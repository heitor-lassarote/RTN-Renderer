#pragma once
#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <crt/host_defines.h>

#include "bsdf.hpp"
#include "camera.hpp"
#include "ray.hpp"
#include "shader_globals.hpp"

namespace rtn
{
struct intersection;

struct shape
{
    bsdf bsdf;

    __device__ __host__
    shape()
        : bsdf(rtn::bsdf{ bsdf_type::none, glm::vec3() })
    {
    }

    __device__ __host__
    explicit shape(rtn::bsdf const& bsdf)
        : bsdf(bsdf)
    {
    }
};

template<typename Shape>
__device__ __host__
shader_globals calculate_shader_globals(
    Shape const&,
    ray const&,
    intersection const&);

template<typename Shape>
__device__ __host__
float surface_area(Shape const&);

struct intersection
{
    float        distance;
    bool         hit;
    shape const* source;

    __device__ __host__
    intersection()
        : distance(99999999)
        , hit(false)
        , source(nullptr)
    {
    }

    __device__ __host__
    intersection(
        float const distance,
        bool const hit,
        shape const* const source)
        : distance(distance)
        , hit(hit)
        , source(source)
    {
    }

    __device__ __host__
    intersection& operator=(intersection other)
    {
        if (&other == this)
        {
            return *this;
        }

        this->distance = other.distance;
        this->hit = other.hit;
        this->source = other.source;
        return *this;
    }
};

template<typename Geometry>
__device__ __host__
intersection intersects(Geometry const&, ray const&);
}

#endif
