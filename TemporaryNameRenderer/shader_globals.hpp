#pragma once
#ifndef SHADER_GLOBALS_HPP
#define SHADER_GLOBALS_HPP

#include <crt/host_defines.h>
#include <glm/glm.hpp>

struct shader_globals
{
    glm::vec3 const point;
    glm::vec3 const normal;
    glm::vec2 const uv;
    glm::vec3 const tangent_u;
    glm::vec3 const tangent_v;
    glm::vec2 const texture_coordinates;
    glm::vec3 const view_direction;

    __device__ __host__
    shader_globals(
        glm::vec3 const point,
        glm::vec3 const normal,
        glm::vec2 const uv,
        glm::vec3 const tangent_u,
        glm::vec3 const tangent_v,
        glm::vec2 const texture_coordinates,
        glm::vec3 const view_direction)
        : point(point)
        , normal(normal)
        , uv(uv)
        , tangent_u(tangent_u)
        , tangent_v(tangent_v)
        , texture_coordinates(texture_coordinates)
        , view_direction(view_direction)
    {
    }
};

#endif
