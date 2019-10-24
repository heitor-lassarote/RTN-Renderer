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
    bsdf bsdf_obj;

    __device__ __host__
    shape()
        : bsdf_obj(bsdf{ bsdf_type::none, glm::vec3() })
    {
    }

    __device__ __host__
    shape(bsdf const& bsdf_obj)
        : bsdf_obj(bsdf_obj)
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
    float  distance;
    size_t index;
    bool   hit;

    __device__ __host__
        intersection()
        : distance(99999999)
        , index(0)
        , hit(false)
    {
    }

    __device__ __host__
    intersection(float const distance, size_t const& index, bool const hit)
        : distance(distance)
        , index(index)
        , hit(hit)
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
        this->index = other.index;
        this->hit = other.hit;
        return *this;
    }
};

template<typename Geometry>
__device__ __host__
intersection intersects(Geometry const&, ray const&);
}

#endif
