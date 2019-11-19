#pragma once
#ifndef SCENE_HPP
#define SCENE_HPP

#include "triangle.hpp"

namespace rtn
{
struct scene
{
    size_t   const         num_triangles;
    triangle const*  const triangles;

    size_t   const         num_lights;
    triangle const** const lights;

    __host__
    scene(
        size_t const num_triangles,
        triangle const* const triangles)
        : num_triangles(num_triangles)
        , triangles(triangles)
        , num_lights(calculate_num_lights())
        , lights(calculate_lights())
    {
    }

    /*__host__
    ~scene()
    {
        check_cuda_errors(cudaFree(this->lights));
    }*/

private:
    __host__
    size_t calculate_num_lights() const
    {
        size_t num_lights = 0;
        for (size_t i = 0; i < this->num_triangles; ++i)
        {
            num_lights += this->triangles[i].bsdf.type == bsdf_type::light;
        }

        return num_lights;
    }

    __host__
    triangle const** calculate_lights() const
    {
        triangle const** lights;
        check_cuda_errors(
            cudaMallocManaged(
                reinterpret_cast<void**>(&lights),
                this->num_lights * sizeof(triangle*)));
        for (size_t i = 0, it_lights = 0; i < num_triangles; ++i)
        {
            if (this->triangles[i].bsdf.type == bsdf_type::light)
            {
                lights[it_lights++] = &this->triangles[i];
            }
        }

        return lights;
    }
};

__device__ __host__
intersection intersects(scene const& s, ray const& r)
{
    intersection hit;
    for (size_t idx = 0; idx < s.num_triangles; ++idx)
    {
        let in = intersects(s.triangles[idx], r);
        if (in.hit && in.distance < hit.distance)
        {
            hit = in;
        }
    }

    return hit;
}

__device__ __host__
bool intersects_triangle(scene const& s, ray const& r, triangle const& t)
{
    intersection hit;
    for (size_t idx = 0; idx < s.num_triangles; ++idx)
    {
        let in = intersects(s.triangles[idx], r);
        if (in.hit && in.distance < hit.distance && &t == &s.triangles[idx])
        {
            hit = in;
        }
    }

    return hit.hit;
}
}

#endif
