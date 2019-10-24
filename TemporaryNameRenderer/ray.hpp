#pragma once
#ifndef RAY_HPP
#define RAY_HPP

#include <crt/host_defines.h>
#include <glm/glm.hpp>

namespace rtn
{
struct ray
{
    glm::vec3 const origin;
    glm::vec3 const direction;

    __device__ __host__
    ray(glm::vec3 const& origin, glm::vec3 const& direction)
        : origin(origin)
        , direction(direction)
    {
    }
};

__device__ __host__
glm::vec3 point(ray const& ray, float const distance)
{
    return ray.origin + distance * ray.direction;
}
}

#endif
