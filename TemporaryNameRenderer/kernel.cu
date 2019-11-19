#include <chrono>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#define OIDN_STATIC_LIB
#include <OpenImageDenoise/oidn.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#include <glm/glm.hpp>

#include "camera.hpp"
#include "let.hpp"
#include "renderer.hpp"
#include "sample_scenes.hpp"
#include "scene.hpp"
#include "triangle.hpp"
#include "utility.hpp"

using namespace rtn;

int main()
{
    std::cout << "Initialize." << std::endl;
    size_t constexpr depth = 4;

    let width = 1200;
    let height = 600;
    let tx = 8;
    let ty = 8;

    let camera_samples = 1;
    let light_samples = 8;

    int const num_pixels = width * height;

    glm::vec3* fb;
    check_cuda_errors(
        cudaMallocManaged(
            reinterpret_cast<void**>(&fb),
            num_pixels * sizeof(glm::vec3)));

    triangle* triangles;
    size_t const num_triangles = 2 * ((5 + 1) + (6) + (6));
    check_cuda_errors(
        cudaMallocManaged(
            reinterpret_cast<void**>(&triangles),
            num_triangles * sizeof(triangle)));

    size_t count = 0;
    simple_scene(triangles, count);

    let box_1_bsdf = bsdf{ bsdf_type::diffuse, glm::vec3(1.0f, 0.0f, 1.0f) };
    let box_1_transform = glm::translate(
        glm::scale(
            glm::rotate(
                glm::mat4(1.0f),
                glm::pi<float>() / 4.0f,
                glm::vec3(0.0f, 1.0f, 0.0f)),
            glm::vec3(0.5f, 0.5f, 0.5f)),
        glm::vec3(0.0f, -1.25f, 0.0f));
    box(triangles, box_1_bsdf, box_1_transform, count);

    let box_2_bsdf = bsdf{ bsdf_type::diffuse, glm::vec3(0.0f, 1.0f, 1.0f) };
    let box_2_transform = glm::translate(
        glm::scale(
            glm::rotate(
                glm::mat4(1.0f),
                3.0f * glm::pi<float>() / 4.0f,
                glm::vec3(0.0f, 1.0f, 0.0f)),
            glm::vec3(0.5f, 1.0f, 0.5f)),
        glm::vec3(2.0f, -0.5f, 0.25f));
    box(triangles, box_2_bsdf, box_2_transform, count);

    scene const scene
    {
        num_triangles,
        triangles,
    };

    camera const camera
    {
        glm::half_pi<float>(),
        static_cast<float const>(width),
        static_cast<float const>(height),
        glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    };

    curandState* rand_states;
    check_cuda_errors(
        cudaMallocManaged(
            reinterpret_cast<void**>(&rand_states),
            num_pixels * sizeof(curandState)));
    glm::vec2* stratified_samples;
    check_cuda_errors(
        cudaMallocManaged(
            reinterpret_cast<void**>(&stratified_samples),
            camera_samples * sizeof(glm::vec2)));

    render_options const render_options(
        width,
        height,
        camera_samples,
        light_samples,
        1.0f,
        1.0f,
        1.0f);

    renderer const renderer(
        render_options,
        camera,
        scene);

    dim3 const blocks(width / tx + 1, height / ty + 1);
    dim3 const threads(tx, ty);

    std::cout << "Raytrace." << std::endl;
    let start = std::chrono::system_clock::now();

    generate_stratified_samples_2d<<<2, 2>>>(stratified_samples, camera_samples);
    check_cuda_errors(cudaGetLastError());
    check_cuda_errors(cudaDeviceSynchronize());

    generate_curand_states<<<blocks, threads>>>(rand_states, width, height);
    check_cuda_errors(cudaGetLastError());
    check_cuda_errors(cudaDeviceSynchronize());

    render<depth><<<blocks, threads>>>(renderer, fb, stratified_samples, rand_states);
    check_cuda_errors(cudaGetLastError());
    check_cuda_errors(cudaDeviceSynchronize());

    let stop = std::chrono::system_clock::now();
    let timer_seconds =
        static_cast<std::chrono::duration<double>>(stop - start).count();
    std::cout << "Took " << timer_seconds << " seconds." << std::endl;

    std::cout << "Denoise." << std::endl;
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color",  fb, oidn::Format::Float3, width, height);
    filter.setImage("output", fb, oidn::Format::Float3, width, height);
    filter.commit();
    filter.execute();

    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
    {
        std::cerr << "OIDN Error: " << errorMessage << std::endl;
    }

    std::cout << "Convert float image to size_t image." << std::endl;
    uint8_t* pixels = new uint8_t[width * height * 3];
    #pragma omp parallel for
    for (int j = height - 1; j >= 0; --j)
    {
        for (int i = 0; i < width; ++i)
        {
            int const pixel_index = j * width + i;
            let ir = static_cast<size_t>(glm::min(255.99f * fb[pixel_index].r, 255.0f));
            let ig = static_cast<size_t>(glm::min(255.99f * fb[pixel_index].g, 255.0f));
            let ib = static_cast<size_t>(glm::min(255.99f * fb[pixel_index].b, 255.0f));
            pixels[3 * pixel_index + 0] = ir;
            pixels[3 * pixel_index + 1] = ig;
            pixels[3 * pixel_index + 2] = ib;
        }
    }

    std::cout << "Save JPG." << std::endl;
    stbi_write_jpg("temporary_name.jpg", width, height, 3, pixels, 100);
    system("temporary_name.jpg");

    std::cout << "Free resources." << std::endl;
    delete[] pixels;
    check_cuda_errors(cudaFree(fb));
    check_cuda_errors(cudaFree(triangles));
    check_cuda_errors(cudaFree(stratified_samples));
    check_cuda_errors(cudaFree(rand_states));
    check_cuda_errors(cudaFree(scene.lights));
    return 0;
}
