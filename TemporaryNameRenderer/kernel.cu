#include <ctime>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#include <glm/glm.hpp>

#include "ray.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

#define let auto const

using namespace renderer_temporary_name;

#define check_cuda_errors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(
    cudaError_t const result,
    char const *const func,
    char const *const file,
    int const line)
{
    if (result)
    {
        std::cerr
            << "CUDA error = " << static_cast<unsigned int>(result)
            << " (" << cudaGetErrorString(result) << ") at "
            << file << ':' << line << " '" << func << "'\n";
        cudaDeviceReset();
        exit(result);
    }
}

template<typename T>
__host__ __device__
T lerp(T const& a, T const& b, float const t)
{
    return (1.0f - t) * a + t * b;
}

__device__
glm::vec3 color(ray const& r)
{
    sphere const s{ bsdf{ bsdf_type::none, glm::vec3() }, glm::vec3(0.0f, 0.0f, -1.0f), 0.5f };
    if (intersects(s, r).hit)
    {
        return glm::vec3(1.0f, 0.0f, 0.0f);
    }

    glm::vec3 const unit_direction = glm::normalize(r.direction);
    float const t = 0.5f * (unit_direction.y + 1.0f);
    return lerp(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.5f, 0.7f, 1.0f), t);
}

__global__
void render(
    glm::vec3 *fb,
    int const max_x,
    int const max_y,
    glm::vec3 const lower_left_corner,
    glm::vec3 const horizontal,
    glm::vec3 const vertical,
    glm::vec3 const origin)
{
    int const i = threadIdx.x + blockIdx.x * blockDim.x;
    int const j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < max_x && j < max_y)
    {
        int const pixel_index = j * max_x + i;
        let u = static_cast<float>(i) / max_x;
        let v = static_cast<float>(j) / max_y;
        ray const r(origin, lower_left_corner + u * horizontal + v * vertical);
        fb[pixel_index] = color(r);
    }
}

__global__
void render(
    glm::vec3* fb,
    sphere* spheres,
    size_t const num_spheres,
    triangle* triangles,
    size_t const num_triangles,
    int const max_x,
    int const max_y,
    glm::vec3 const lower_left_corner,
    glm::vec3 const horizontal,
    glm::vec3 const vertical,
    camera const camera)
{
    int const i = threadIdx.x + blockIdx.x * blockDim.x;
    int const j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < max_x && j < max_y)
    {
        int const pixel_index = j * max_x + i;
        let u = static_cast<float>(i) / max_x;
        let v = static_cast<float>(j) / max_y;
        ray const r(
            camera.position,
            lower_left_corner + u * horizontal + v * vertical);

        shape* shape = nullptr;
        intersection hit;
        for (size_t idx = 0; idx < num_spheres; ++idx)
        {
            let in = intersects(spheres[idx], r);
            if (in.hit && in.distance < hit.distance)
            {
                shape = spheres;
                hit = in;
            }
        }

        for (size_t idx = 0; idx < num_triangles; ++idx)
        {
            let in = intersects(triangles[idx], r);
            if (in.hit && in.distance < hit.distance)
            {
                shape = triangles;
                hit = in;
            }
        }

        if (hit.hit)
        {
            fb[pixel_index] = shape->bsdf_obj.color;
        }
    }
}

int main()
{
    int const nx = 1200;
    int const ny = 600;
    int const tx = 8;
    int const ty = 8;

    int const num_pixels = nx * ny;
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

    clock_t const start = clock();
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(
        fb,
        spheres,
        num_spheres,
        triangles,
        num_triangles,
        nx,
        ny,
        glm::vec3(-2.0f, -1.0f, -1.0f),
        glm::vec3( 4.0f,  0.0f,  0.0f),
        glm::vec3( 0.0f,  2.0f,  0.0f),
        camera{ glm::vec3(0.0f,  0.0f,  0.0f) });
    check_cuda_errors(cudaGetLastError());
    check_cuda_errors(cudaDeviceSynchronize());
    clock_t const stop = clock();
    double const timer_seconds =
        static_cast<double>(stop - start) / CLOCKS_PER_SEC;
    std::cout << "Took " << timer_seconds << " seconds\n";

    uint8_t* pixels = new uint8_t[nx * ny * 3];
    for (int j = ny - 1; j >= 0; --j)
    {
        #pragma omp parallel for
        for (int i = 0; i < nx; ++i)
        {
            int const pixel_index = j * nx + i;
            let ir = static_cast<int>(255.99f * fb[pixel_index].r);
            let ig = static_cast<int>(255.99f * fb[pixel_index].g);
            let ib = static_cast<int>(255.99f * fb[pixel_index].b);
            pixels[3 * pixel_index + 0] = ir;
            pixels[3 * pixel_index + 1] = ig;
            pixels[3 * pixel_index + 2] = ib;
        }
    }

    stbi_write_jpg("temporary_name.jpg", nx, ny, 3, pixels, 100);
    system("temporary_name.jpg");

    delete[] pixels;
    check_cuda_errors(cudaFree(fb));
    check_cuda_errors(cudaFree(spheres));
    check_cuda_errors(cudaFree(triangles));
    return 0;
}
