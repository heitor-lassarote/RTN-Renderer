#pragma once
#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>

#include "let.hpp"
#include "scene.hpp"

namespace rtn
{
struct render_options
{
    int   const width;
    int   const height;
    int   const maximum_depth;
    int   const camera_samples;
    int   const light_samples;
    float const filter_width;
    float const gamma;
    float const exposure;

    __device__ __host__
    render_options(
        int const width,
        int const height,
        int const maximum_depth,
        int const camera_samples,
        int const light_samples,
        float const filter_width,
        float const gamma,
        float const exposure)
        : width(width)
        , height(height)
        , maximum_depth(maximum_depth)
        , camera_samples(camera_samples)
        , light_samples(light_samples)
        , filter_width(filter_width)
        , gamma(gamma)
        , exposure(exposure)
    {
    }
};

struct renderer
{
    render_options const options;
    camera         const camera;
    scene          const scene;

    __device__ __host__
    renderer(
        render_options const options,
        rtn::camera const camera,
        rtn::scene const scene)
        : options(options)
        , camera(camera)
        , scene(scene)
    {
    }
};

glm::vec3* compute_direct_illumination(
    bsdf const* const,
    shader_globals const* const) = delete;
glm::vec3* compute_indirect_illumination(
    bsdf const* const,
    shader_globals const* const,
    int const depth) = delete;

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
void render_init(glm::vec2* samples, size_t const size)
{
    let x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= size)
    {
        return;
    }

    curandState rng;
    curand_init(42 + x, 0, 0, &rng);
    stratified_sample(size, samples, rng);
}

__device__ __host__
glm::vec3 trace(renderer const renderer, ray const ray, int const depth)
{
    let i = intersects(renderer.scene, ray);
    return i.hit ? i.source->bsdf.color : glm::vec3(0.0f);
}

__global__
void render(
    renderer const renderer,
    glm::vec3* fb,
    glm::vec2* samples)
{
    let options = renderer.options;

    let x = threadIdx.x + blockIdx.x * blockDim.x;
    let y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= options.width || y >= options.height)
    {
        return;
    }

    let pixel_index = y * options.width + x;
    glm::vec3& color = fb[pixel_index];
    color = glm::vec3(0.0f);
    float total_weight = 0.0f;

    for (size_t k = 0; k < options.camera_samples; ++k)
    {
        let s_sample = samples[k];
        let sample = (s_sample - glm::vec2(0.5f)) * options.filter_width;
        let sample_dof = s_sample;

        let r = generate_ray(renderer.camera, x, y, sample, sample_dof);
        let weight = gaussian_2d(sample, options.filter_width);

        color += trace(renderer, r, 0) * weight;
        total_weight += weight;
    }

    color /= total_weight;
    color = glm::clamp(
        gamma(exposure(color, options.exposure), options.gamma),
        0.0f,
        1.0f);
}
}

#endif
