#pragma once
#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <utility>

#include <crt/host_defines.h>
#include <glm/glm.hpp>

#include "bsdf.hpp"
#include "shader_globals.hpp"
#include "vertex.hpp"

namespace rtn
{
struct intersection;

struct triangle
{
    bsdf bsdf;
    vertex v0;
    vertex v1;
    vertex v2;

    __device__ __host__
    triangle()
        : bsdf(bsdf_type::none, glm::vec3())
        , v0(glm::vec3(), glm::vec3(), glm::vec2())
        , v1(glm::vec3(), glm::vec3(), glm::vec2())
        , v2(glm::vec3(), glm::vec3(), glm::vec2())
    {
    }

    __device__ __host__
    triangle(rtn::bsdf const& bsdf, vertex v0, vertex v1, vertex v2)
        : bsdf(bsdf)
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

        this->bsdf = other.bsdf;
        this->v0 = other.v0;
        this->v1 = other.v1;
        this->v2 = other.v2;
        return *this;
    }
};

struct intersection
{
    float           distance;
    bool            hit;
    triangle const* source;

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
            triangle const* const source)
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
        return { t0, true, &t };
    }
    else
    {
        // This means that there is a line intersection but not a ray
        // intersection.
        return {};
    }
}

__device__ __host__
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

__device__ __host__
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

__device__ __host__
glm::vec3 barycenter_weight(
    glm::vec3 const& p0,
    glm::vec3 const& p1,
    glm::vec3 const& p2,
    glm::vec3 const& p)
{
    let d00 = glm::dot(p0, p0);
    let d01 = glm::dot(p0, p1);
    let d11 = glm::dot(p1, p1);
    let d20 = glm::dot(p2, p0);
    let d21 = glm::dot(p2, p1);

    let inverse_d = d00 * d11 - d01 * d01;

    let v = (d11 * d20 - d01 * d21) / inverse_d;
    let w = (d00 * d21 - d01 * d20) / inverse_d;

    return glm::vec3(1.0f - v - w, v, w);
}

__device__ __host__
shader_globals calculate_shader_globals(
    triangle const& t,
    ray const& r,
    intersection const& i)
{
    let p0 = t.v0.position;
    let p1 = t.v1.position;
    let p2 = t.v2.position;
    let n0 = t.v0.normal;
    let n1 = t.v1.normal;
    let n2 = t.v2.normal;
    let t0 = t.v0.uv;
    let t1 = t.v1.uv;
    let t2 = t.v2.uv;

    let position = point(r, i.distance);
    let bc = barycenter_weight(p0, p1, p2, position);
    let normal = normalize((bc.x * n0) + (bc.y * n1) + (bc.z * n2));

    let texture_coordinates = (t0 * bc.x) + (t1 * bc.y) + (t2 * bc.z);

    let tangent_u = glm::abs(normal.x) >= glm::abs(normal.y)
        ? glm::normalize(glm::vec3(normal.z, 0.0f, -normal.x))
        : glm::normalize(glm::vec3(0.0f, -normal.z, normal.y));
    let tangent_v = glm::cross(normal, tangent_u);

    let view_dir = -r.direction;

    let uv = glm::vec2(bc.x, bc.y);

    return
    {
        position,
        normal,
        uv,
        tangent_u,
        tangent_v,
        texture_coordinates,
        view_dir,
    };
}

__device__ __host__
float surface_area(triangle const& t)
{
    glm::vec3 const e1 = t.v1.position - t.v0.position;
    glm::vec3 const e2 = t.v2.position - t.v0.position;
    return 0.5f * length(cross(e1, e2));
}
}

#endif
