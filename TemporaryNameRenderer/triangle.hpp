#pragma once
#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <utility>

#include <crt/host_defines.h>
#include <glm/glm.hpp>

#include "shape.hpp"
#include "vertex.hpp"

namespace renderer_temporary_name
{
struct triangle : public shape
{
    vertex v0;
    vertex v1;
    vertex v2;

    __device__ __host__
    triangle(bsdf const& bsdf_obj, vertex v0, vertex v1, vertex v2)
        : shape(bsdf_obj)
        , v0(std::move(v0))
        , v1(std::move(v1))
        , v2(std::move(v2))
    {
    }

    __device__ __host__
    triangle& operator=(triangle other)
    {
        if (&other == this)
        {
            return *this;
        }

        this->bsdf_obj = other.bsdf_obj;
        this->v0 = other.v0;
        this->v1 = other.v1;
        this->v2 = other.v2;
        return *this;
    }
};

template<>
__device__ __host__
intersection intersects(triangle const& t, ray const& ray)
{
    float const epsilon = 0.0000001f;

    glm::vec3 const v0 = t.v0.position;
    glm::vec3 const v1 = t.v1.position;
    glm::vec3 const v2 = t.v2.position;
    glm::vec3 const edge1 = v1 - v0;
    glm::vec3 const edge2 = v2 - v0;
    glm::vec3 const h = cross(ray.direction, edge2);
    float const a = dot(edge1, h);

    if (a > -epsilon && a < epsilon)
    {
        return {}; // This ray is parallel to this triangle.
    }

    float const f = 1.0f / a;
    glm::vec3 const s = ray.origin - v0;
    float const u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f)
    {
        return {};
    }

    glm::vec3 const q = cross(s, edge1);
    float const v = f * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f)
    {
        return {};
    }

    // At this stage we can compute t to find out where the intersection point
    // is on the line.
    float const t0 = f * dot(edge2, q);
    if (t0 > epsilon && t0 < 1.0f / epsilon) // ray intersection
    {
        return { t0, 0, true };
    }
    else
    {
        // This means that there is a line intersection but not a ray
        // intersection.
        return {};
    }
}

glm::vec2 barycenter_to_vertex(
    glm::vec2 const& t0,
    glm::vec2 const& t1,
    glm::vec2 const& t2,
    glm::vec3 const& bc)
{
    return glm::vec2(
        (bc.x * t0.x) + (bc.y * t1.x) + (bc.z * t2.x),
        (bc.x * t0.y) + (bc.y * t1.y) + (bc.z * t2.y));
}

glm::vec3 barycenter_to_vertex(
    glm::vec3 const& t0,
    glm::vec3 const& t1,
    glm::vec3 const& t2,
    glm::vec3 const& bc)
{
    return glm::vec3(
        (bc.x * t0.x) + (bc.y * t1.x) + (bc.z * t2.x),
        (bc.x * t0.y) + (bc.y * t1.y) + (bc.z * t2.y),
        (bc.x * t0.z) + (bc.y * t1.z) + (bc.z * t2.z));
}

glm::vec3 barycenter_weight(
    glm::vec3 const& p0,
    glm::vec3 const& p1,
    glm::vec3 const& p2,
    glm::vec3 const& p)
{
    glm::vec3 const u = cross(
        glm::vec3(p2.x - p0.x, p1.x - p0.x, p0.x - p.x),
        glm::vec3(p2.y - p0.y, p1.y - p0.y, p0.y - p.y));

    // pts and p have integer values as coordinates, so |u.Z| < 1 means
    // u.Z is 0, that means triangle is degenerate, in this case return
    // something with negative coordinates.
    float const epsilon = 1e-2f;
    if (abs(u.z) > epsilon)
    {
        return glm::vec3(
            1.0f - ((u.x + u.y) / u.z),
            u.y / u.z,
            u.x / u.z);
    }

    return glm::vec3(-1.0f, 1.0f, 1.0f);
}

template<>
__device__ __host__
shader_globals calculate_shader_globals(
    triangle const& t,
    ray const& r,
    intersection const& i)
{
    glm::vec3 const p0 = t.v0.position;
    glm::vec3 const p1 = t.v1.position;
    glm::vec3 const p2 = t.v2.position;
    glm::vec3 const n0 = t.v0.normal;
    glm::vec3 const n1 = t.v1.normal;
    glm::vec3 const n2 = t.v2.normal;
    glm::vec2 const t0 = t.v0.uv;
    glm::vec2 const t1 = t.v1.uv;
    glm::vec2 const t2 = t.v2.uv;

    glm::vec3 const position = point(r, i.distance);
    glm::vec3 const barycenter = barycenter_weight(p0, p1, p2, position);
    glm::vec3 const normal = normalize(barycenter.x * n0 + barycenter.z * n1 + barycenter.y * n2);

    glm::vec2 const uv = barycenter_to_vertex(t0, t1, t2, barycenter);

    // n.b.: the tangent should be defined in terms of the normal and the
    // Gram-Schmidt process.
    glm::vec3 const tangent_u = p1 - p0;
    glm::vec3 const tangent_v = p2 - p0;

    // n.b.: this might use the camera position instead of ray origin in the
    // future.
    glm::vec3 const view_dir = normalize(position - r.origin);

    glm::vec3 const light_direction(0.0f);
    glm::vec3 const light_point(0.0f);
    glm::vec3 const light_normal(0.0f);

    return
    {
        position,
        normal,
        uv,
        tangent_u,
        tangent_v,
        view_dir,
        light_direction,
        light_point,
        light_normal,
    };
}

template<>
__device__ __host__
float surface_area(triangle const& t)
{
    glm::vec3 const e1 = t.v1.position - t.v0.position;
    glm::vec3 const e2 = t.v2.position - t.v0.position;
    return 0.5f * length(cross(e1, e2));
}
}

#endif
