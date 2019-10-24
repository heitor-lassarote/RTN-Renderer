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
}

#endif
