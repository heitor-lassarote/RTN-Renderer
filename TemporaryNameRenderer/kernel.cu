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
#include "ray.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

using namespace rtn;

__global__
void render_init(
    curandState* rand_state,
    int const max_x,
    int const max_y)
{
    let x = threadIdx.x + blockIdx.x * blockDim.x;
    let y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= max_x || y >= max_y)
    {
        return;
    }

    let pixel_index = y * max_x + x;
    curand_init(42 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__
void render(
    glm::vec3* fb,
    curandState* rand_state,
    sphere* spheres,
    size_t const num_spheres,
    triangle* triangles,
    size_t const num_triangles,
    int const max_x,
    int const max_y,
    camera const camera,
    size_t const camera_samples)
{
    let x = threadIdx.x + blockIdx.x * blockDim.x;
    let y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= max_x || y >= max_y)
    {
        return;
    }

    let pixel_index = y * max_x + x;
    fb[pixel_index] = glm::vec3(0.0f);

    for (size_t i = 0; i < camera_samples; ++i)
    {
        glm::vec2 sample(
            curand_uniform(&rand_state[pixel_index]),
            curand_uniform(&rand_state[pixel_index]));

        let r = generate_ray(camera, x, y, sample);

        shape* shape_array = nullptr;
        intersection hit;
        for (size_t idx = 0; idx < num_spheres; ++idx)
        {
            let in = intersects(spheres[idx], r);
            if (in.hit && in.distance < hit.distance)
            {
                shape_array = spheres;
                hit = in;
            }
        }

        for (size_t idx = 0; idx < num_triangles; ++idx)
        {
            let in = intersects(triangles[idx], r);
            if (in.hit && in.distance < hit.distance)
            {
                shape_array = triangles;
                hit = in;
            }
        }

        if (hit.hit)
        {
            fb[pixel_index] += shape_array[hit.index].bsdf_obj.color;
        }
    }

    fb[pixel_index] /= camera_samples;
}

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

    camera const cam
    {
        glm::half_pi<float>(),
        static_cast<float const>(width),
        static_cast<float const>(height),
        glm::mat4(1.0f),
    };

    curandState* rand_state;
    check_cuda_errors(
        cudaMallocManaged(
            reinterpret_cast<void**>(&rand_state),
            num_pixels * sizeof(curandState)));

    let start = std::chrono::system_clock::now();

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    render_init<<<blocks, threads>>>(
        rand_state,
        width,
        height);
    check_cuda_errors(cudaGetLastError());
    check_cuda_errors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(
        fb,
        rand_state,
        spheres,
        num_spheres,
        triangles,
        num_triangles,
        width,
        height,
        cam,
        camera_samples);
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
    return 0;
}
