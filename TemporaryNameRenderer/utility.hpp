#pragma once
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <glm/glm.hpp>
#include <glm/vec2.hpp>

#include "let.hpp"

namespace rtn
{
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

__device__ __host__
glm::vec2 concentric_sample_disk(glm::vec2 const& sample)
{
    glm::vec2 const offset = 2.0f * sample - glm::vec2(1.0f, 1.0f);

    if (offset.x == 0.0f && offset.y == 0.0f)
    {
        return glm::vec2(0.0f, 0.0f);
    }

    let pi = 4.0f * glm::atan(1.0f);
    if (glm::abs(offset.x) > glm::abs(offset.y))
    {
        let radius = offset.x;
        let theta = 0.25f * pi * offset.y / offset.x;
        return radius * glm::vec2(glm::cos(theta), glm::sin(theta));
    }
    else
    {
        let radius = offset.y;
        let theta = pi * (0.5f - 0.25f * offset.x / offset.y);
        return radius * glm::vec2(glm::cos(theta), glm::sin(theta));
    }
}

__device__
glm::vec2 uniform_random_2d(curandState& rng)
{
    return glm::vec2(
        curand_uniform(&rng),
        curand_uniform(&rng));
}

__device__
void stratified_sample(
    int const samples,
    glm::vec2* points,
    curandState& rng)
{
    size_t const size = glm::sqrt(static_cast<float>(samples));
    for (size_t x = 0; x < size; ++x)
    {
        for (size_t y = 0; y < size; ++y)
        {
            glm::vec2 const offset(x, y);
            size_t const idx = y * size + x;
            points[idx] = (offset + uniform_random_2d(rng))
                / static_cast<float>(size);
        }
    }
}

__device__ __host__
glm::vec3 gamma(glm::vec3 const color, float const value)
{
    float const inverse_gamma = 1.0f / value;
    return glm::vec3(
        glm::pow(color.r, inverse_gamma),
        glm::pow(color.g, inverse_gamma),
        glm::pow(color.b, inverse_gamma));
}

__device__ __host__
glm::vec3 exposure(glm::vec3 const color, float const value)
{
    float const power = glm::pow(2.0f, value);
    return glm::vec3(
        glm::pow(color.r, power),
        glm::pow(color.g, power),
        glm::pow(color.b, power));
}

__device__ __host__
float gaussian(float const sample, float const r)
{
    return glm::max(glm::exp(-sample * sample) - glm::exp(-r * r), 0.0f);
}

__device__ __host__
float gaussian_2d(glm::vec2 const& sample, float const width)
{
    float const r = width / 2.0f;
    return gaussian(sample.x, r) * gaussian(sample.y, r);
}
}

#endif
