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
#include "triangle.hpp"

namespace rtn
{
struct render_options
{
    int   const width;
    int   const height;
    int   const camera_samples;
    int   const light_samples;
    float const filter_width;
    float const gamma;
    float const exposure;

    __device__ __host__
    render_options(
        int const width,
        int const height,
        int const camera_samples,
        int const light_samples,
        float const filter_width,
        float const gamma,
        float const exposure)
        : width(width)
        , height(height)
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

__global__
void generate_curand_states(
    curandState* rand_state,
    int const width,
    int const height)
{
    let x = threadIdx.x + blockIdx.x * blockDim.x;
    let y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
    {
        return;
    }

    let pixel_index = y * width + x;
    curand_init(42 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__
void generate_stratified_samples_2d(glm::vec2* samples, size_t const size)
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

__global__
void generate_samples(float* samples, size_t const size)
{
    let x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= size)
    {
        return;
    }

    curandState rng;
    curand_init(69 + x, 0, 0, &rng);
    samples[x] = curand_uniform(&rng);
}

__device__
glm::vec3 compute_direct_illumination(
    renderer const& renderer,
    bsdf const& bsdf,
    shader_globals const& globals,
    curandState& rand_state)
{
    let fr = bsdf.color * glm::one_over_pi<float>();

    let N = renderer.options.light_samples;
    let num_lights = renderer.scene.num_lights;

    let w_0 = globals.view_direction;
    let n = globals.normal;
    let x = globals.point;

    glm::vec3 result(0.0f);
    for (size_t i = 0; i < N; ++i)
    {
        let random_idx = glm::min(
            static_cast<size_t>(num_lights * curand_uniform(&rand_state)),
            num_lights - 1);

        let light = renderer.scene.lights[random_idx];

        let lp0 = light->v0.position;
        let lp1 = light->v1.position;
        let lp2 = light->v2.position;
        let ln0 = light->v0.normal;
        let ln1 = light->v1.normal;
        let ln2 = light->v2.normal;

        let bc = low_discrepancy_sample_triangle(
            1.0f - curand_uniform(&rand_state));

        let x_ = (bc.x * lp0) + (bc.y * lp1) + (bc.z * lp2);
        let w_i = glm::normalize(x_ - x);
        let length_w_i = glm::length(x_ - x);

        let n_ = (bc.x * ln0) + (bc.y * ln1) + (bc.z * ln2);
        let l_i = light->bsdf.color;

        let v = intersects_triangle(renderer.scene, { x, w_i }, *light);

        let pdf = 1.0f / (surface_area(*light) * num_lights);
        result += fr / pdf * l_i * glm::dot(w_i, n)
                * glm::dot(-w_i, n_) / (length_w_i * length_w_i)
                * static_cast<float>(v);
    }

    return result / static_cast<float>(N);
}

template<size_t Depth>
__device__
constexpr glm::vec3 trace(
    renderer const& renderer,
    ray const& ray,
    int const depth,
    curandState& rand_state);

template<size_t Depth>
__device__
glm::vec3 compute_indirect_illumination(
    renderer const& renderer,
    bsdf const& bsdf,
    shader_globals const& globals,
    curandState& rand_state)
{
    let fr = bsdf.color * glm::one_over_pi<float>();

    let N = renderer.options.light_samples;

    let w_0 = globals.view_direction;
    let n = globals.normal;
    let x = globals.point;

    glm::vec3 result(0.0f);
    for (size_t i = 0; i < N; ++i)
    {
        let transformation = glm::mat4(
            glm::vec4(globals.tangent_u, 0.0f),
            glm::vec4(n, 0.0f),
            glm::vec4(globals.tangent_v, 0.0f),
            glm::vec4(glm::vec3(0.0f), 1.0f));

        let sample = glm::vec2(
            curand_uniform(&rand_state),
            curand_uniform(&rand_state));
        let x_ = glm::vec3(
            transformation * glm::vec4(uniform_sample_hemisphere(sample), 0.0f));
        let w_i = glm::normalize(x_ - x);

        let l_i = trace<Depth - 1>(
            renderer,
            { x + 0.01f * n, w_i },
            rand_state);

        let pdf = glm::one_over_two_pi<float>();
        result += fr / pdf * l_i * glm::dot(w_i, n);
    }

    return result / static_cast<float>(N);
}

template<size_t Depth>
__device__
glm::vec3 trace(
    renderer const& renderer,
    ray const& ray,
    curandState& rand_state)
{
    let i = intersects(renderer.scene, ray);
    //return i.hit ? i.source->bsdf.color : glm::vec3(0.0f);

    if (i.hit)
    {
        let bsdf = i.source->bsdf;
        if (bsdf.type == bsdf_type::light)
        {
            return bsdf.color;
        }

        let globals = calculate_shader_globals(
            *i.source,
            ray,
            i);

        let direct = compute_direct_illumination(
            renderer,
            bsdf,
            globals,
            rand_state);
        let indirect = compute_indirect_illumination<Depth>(
            renderer,
            bsdf,
            globals,
            rand_state);

        return indirect + direct;
    }
    else
    {
        return glm::vec3(0.0f);
    }
}

template<>
__device__
glm::vec3 trace<0>(
    renderer const& renderer,
    ray const& ray,
    curandState& rand_state)
{
    return glm::vec3(0.0f);
}

template<size_t Depth>
__global__
void render(
    renderer const renderer,
    glm::vec3* fb,
    glm::vec2* stratified_samples,
    curandState* rand_states)
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

    if (options.light_samples == 0)
    {
        return;
    }

    float total_weight = 0.0f;
    for (size_t k = 0; k < options.camera_samples; ++k)
    {
        let s_sample = stratified_samples[k];
        let sample = (s_sample - glm::vec2(0.5f)) * options.filter_width;
        let sample_dof = s_sample;

        let r = generate_ray(renderer.camera, x, y, sample, sample_dof);
        let weight = gaussian_2d(sample, options.filter_width);

        color += trace<Depth>(renderer, r, rand_states[pixel_index]) * weight;
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
