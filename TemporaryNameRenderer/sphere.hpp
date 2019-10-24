#pragma once
#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <crt/host_defines.h>
#include <glm/glm.hpp>

#include "ray.hpp"
#include "shape.hpp"

namespace rtn
{
struct sphere : public shape
{
    glm::vec3 position;
    float     radius;

    __device__ __host__
    sphere(bsdf const& bsdf, glm::vec3 const& position, float const radius)
        : shape(bsdf)
        , position(position)
        , radius(radius)
    {
    }

    __device__ __host__
    sphere& operator=(sphere other)
    {
        if (&other == this)
        {
            return *this;
        }

        this->bsdf_obj = other.bsdf_obj;
        this->position = other.position;
        this->radius = other.radius;
        return *this;
    }
};

template<>
__device__ __host__
intersection intersects(sphere const& s, ray const& r)
{
    glm::vec3 const oc = r.origin - s.position;
    float const a = glm::dot(r.direction, r.direction);
    float const b = 2.0f * glm::dot(oc, r.direction);
    float const c = glm::dot(oc, oc) - s.radius * s.radius;
    float const discriminant = b * b - 4.0f * a * c;
    if (discriminant > 0.0f)
    {
        float const t0 = (-b + sqrt(discriminant)) / (2.0f * a);
        float const t1 = (-b - sqrt(discriminant)) / (2.0f * a);
        float const t = glm::min(t0, t1);

        // Is the smallest t necessarily the smallest distance?
        return { t, 0, true };
    }
    else
    {
        return {};
    }
}

template<>
__device__ __host__
shader_globals calculate_shader_globals(
    sphere const& s,
    ray const& r,
    intersection const& i)
{
    glm::vec3 const position = point(r, i.distance);
    glm::vec3 const normal = normalize(position - s.position);

    float const theta = glm::atan(normal.z, normal.x);
    float const phi = glm::acos(normal.y);
    glm::vec3 const tangent_u(
        glm::cos(theta),
        0.0f,
        -glm::sin(theta));
    glm::vec3 const tangent_v = glm::vec3(
        glm::sin(theta) * glm::cos(phi),
        -glm::sin(phi),
        glm::cos(theta) * glm::cos(phi));

    glm::vec2 const uv(
        theta / (2.0f * glm::pi<float>()),
        phi / glm::pi<float>());

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
float surface_area(sphere const& s)
{
    return 4.0f * glm::pi<float>() * s.radius * s.radius;
}
}

#endif
