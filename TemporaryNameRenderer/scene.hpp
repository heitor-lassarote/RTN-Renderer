#pragma once
#ifndef SCENE_HPP
#define SCENE_HPP

#include "sphere.hpp"
#include "triangle.hpp"

namespace rtn
{
struct scene
{
    sphere   const* const spheres;
    size_t   const        num_spheres;
    triangle const* const triangles;
    size_t   const        num_triangles;

    __device__ __host__
    scene(
        sphere const* const spheres,
        size_t const num_spheres,
        triangle const* const triangles,
        size_t const num_triangles)
        : spheres(spheres)
        , num_spheres(num_spheres)
        , triangles(triangles)
        , num_triangles(num_triangles)
    {
    }
};

template<typename Shape>
__device__ __host__
intersection intersects_shapes(
    Shape const* const ss,
    size_t const size,
    ray const& r)
{
    intersection hit;
    for (size_t idx = 0; idx < size; ++idx)
    {
        let in = intersects(ss[idx], r);
        if (in.hit && in.distance < hit.distance)
        {
            hit = in;
            hit.source = ss;
        }
    }

    return hit;
}

template<>
__device__ __host__
intersection intersects(scene const& s, ray const& r)
{
    intersection hit;
    let hits = intersects_shapes(s.spheres, s.num_spheres, r);
    if (hits.hit)
    {
        hit = hits;
    }

    let hitt = intersects_shapes(s.triangles, s.num_triangles, r);
    if (hitt.hit && hitt.distance < hit.distance)
    {
        hit = hitt;
    }

    return hit;
}
}

#endif
