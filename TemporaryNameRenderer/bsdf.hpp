#pragma once
#ifndef BSDF_HPP
#define BSDF_HPP

#include <crt/host_defines.h>
#include <glm/glm.hpp>

namespace rtn
{
enum class bsdf_type
{
    none,
    light,
    diffuse,
    specular,
};

struct bsdf
{
    bsdf_type type;
    glm::vec3 color;

    __device__ __host__
    bsdf(bsdf_type const& type, glm::vec3 const& color)
        : type(type)
        , color(color)
    {
    }

    __device__ __host__
    bsdf& operator=(bsdf const& other)
    {
        if (&other == this)
        {
            return *this;
        }

        this->type = other.type;
        this->color = other.color;
        return *this;
    }
};
}

#endif
