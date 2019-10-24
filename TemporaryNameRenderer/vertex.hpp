#pragma once
#ifndef VERTEX_HPP
#define VERTEX_HPP

#include <crt/host_defines.h>
#include <glm/glm.hpp>

namespace rtn
{
struct vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;

    __device__ __host__
    vertex(
        glm::vec3 const& position,
        glm::vec3 const& normal,
        glm::vec2 const& uv)
        : position(position)
        , normal(normal)
        , uv(uv)
    {
    }

    __device__ __host__
    vertex& operator=(vertex other)
    {
        if (&other == this)
        {
            return *this;
        }

        this->position = other.position;
        this->normal = other.normal;
        this->uv = other.uv;
        return *this;
    }
};
}

#endif
