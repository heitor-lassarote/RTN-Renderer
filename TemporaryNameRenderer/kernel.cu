#include <chrono>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#include <glm/glm.hpp>

#include "let.hpp"
#include "renderer.hpp"
#include "sphere.hpp"
#include "triangle.hpp"
#include "scene.hpp"

using namespace rtn;

int main()
{
    let width = 1200;
    let height = 600;
    let tx = 8;
    let ty = 8;

    let camera_samples = 1024;

    int const num_pixels = width * height;
    size_t const fb_size = num_pixels * sizeof(glm::vec3);

    glm::vec3* fb;
    check_cuda_errors(
        cudaMallocManaged(reinterpret_cast<void**>(&fb), fb_size));

    sphere* spheres;
    size_t const num_spheres = 1;
    check_cuda_errors(
        cudaMallocManaged(
            reinterpret_cast<void**>(&spheres),
            num_spheres * sizeof(sphere)));

    spheres[0] = sphere
    {
        bsdf{ bsdf_type::none, glm::vec3(0.0f, 0.0f, 1.0f) },
        glm::vec3(0.0f, 0.0f, -2.0f),
        1.0f,
    };

    triangle* triangles;
    size_t const num_triangles = 4;
    check_cuda_errors(
        cudaMallocManaged(
            reinterpret_cast<void**>(&triangles),
            num_triangles * sizeof(triangle)));

    triangles[0] = triangle
    {
        bsdf{ bsdf_type::none, glm::vec3(0.0f, 1.0f, 0.0f) },
        vertex
        {
            glm::vec3(-0.5f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.5f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.0f, 1.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.5f, 1.0f)
        },
    };

    triangles[1] = triangle
    {
        bsdf{ bsdf_type::none, glm::vec3(0.0f, 1.0f, 0.0f) },
        vertex
        {
            glm::vec3(0.5f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(-0.5f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.0f, -1.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.5f, 1.0f)
        },
    };

    triangles[2] = triangle
    {
        bsdf{ bsdf_type::none, glm::vec3(0.0f, 1.0f, 0.0f) },
        vertex
        {
            glm::vec3(0.0f, 0.5f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.0f, -0.5f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(1.0f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.5f, 1.0f)
        },
    };

    triangles[3] = triangle
    {
        bsdf{ bsdf_type::none, glm::vec3(0.0f, 1.0f, 0.0f) },
        vertex
        {
            glm::vec3(0.0f, -0.5f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.0f, 0.5f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(-1.0f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.5f, 1.0f)
        },
    };

    scene const scene
    {
        spheres,
        num_spheres,
        triangles,
        num_triangles
    };

    camera const camera
    {
        glm::half_pi<float>(),
        static_cast<float const>(width),
        static_cast<float const>(height),
        glm::mat4(1.0f),
    };

    /*curandState* rand_state;
    check_cuda_errors(
        cudaMallocManaged(
            reinterpret_cast<void**>(&rand_state),
            camera_samples * sizeof(curandState)));*/
    glm::vec2* samples;
    check_cuda_errors(
        cudaMallocManaged(
            reinterpret_cast<void**>(&samples),
            camera_samples * sizeof(glm::vec2)));

    render_options const render_options(
        width,
        height,
        1,
        camera_samples,
        1,
        1.0f,
        1.0f,
        1.0f);

    renderer const renderer(
        render_options,
        camera,
        scene);

    dim3 const blocks(width / tx + 1, height / ty + 1);
    dim3 const threads(tx, ty);

    let start = std::chrono::system_clock::now();

    render_init<<<32, 32>>>(samples, camera_samples);
    check_cuda_errors(cudaGetLastError());
    check_cuda_errors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(renderer, fb, samples);
    check_cuda_errors(cudaGetLastError());
    check_cuda_errors(cudaDeviceSynchronize());

    let stop = std::chrono::system_clock::now();
    let timer_seconds =
        static_cast<std::chrono::duration<double>>(stop - start).count();
    std::cout << "Took " << timer_seconds << " seconds\n";

    uint8_t* pixels = new uint8_t[width * height * 3];
    for (int j = height - 1; j >= 0; --j)
    {
        #pragma omp parallel for
        for (int i = 0; i < width; ++i)
        {
            int const pixel_index = j * width + i;
            let ir = static_cast<int>(255.99f * fb[pixel_index].r);
            let ig = static_cast<int>(255.99f * fb[pixel_index].g);
            let ib = static_cast<int>(255.99f * fb[pixel_index].b);
            pixels[3 * pixel_index + 0] = ir;
            pixels[3 * pixel_index + 1] = ig;
            pixels[3 * pixel_index + 2] = ib;
        }
    }

    stbi_write_jpg("temporary_name.jpg", width, height, 3, pixels, 100);
    system("temporary_name.jpg");

    delete[] pixels;
    check_cuda_errors(cudaFree(fb));
    check_cuda_errors(cudaFree(spheres));
    check_cuda_errors(cudaFree(triangles));
    check_cuda_errors(cudaFree(samples));
    return 0;
}
