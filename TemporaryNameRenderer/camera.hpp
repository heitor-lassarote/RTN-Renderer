#pragma once
#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <crt/host_defines.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec3.hpp>

#include "let.hpp"
#include "ray.hpp"
#include "utility.hpp"

namespace rtn
{
struct camera
{
    float const fov;
    float const width;
    float const height;

    union
    {
        glm::mat4 const world_matrix;

        struct
        {
            glm::vec4 const i;
            glm::vec4 const j;
            glm::vec4 const k;
            glm::vec4 const p;
        };
    };

    float const focal_length;
    float const f_number;

    __device__ __host__
    camera(
        float const fov_rad,
        float const width,
        float const height,
        glm::mat4 const& world_matrix,
        float const focal_length = 1.0f,
        float const f_number = 2.0f)
        : fov(fov_rad)
        , width(width)
        , height(height)
        , world_matrix(world_matrix)
        , focal_length(focal_length)
        , f_number(f_number)
    {
    }
};

__device__ __host__
camera look_at(
    camera const& c,
    glm::vec3 const& position,
    glm::vec3 const& target,
    glm::vec3 const& up)
{
    let look = glm::lookAt(position, target, up);
    return
    {
        c.fov,
        c.width,
        c.height,
        std::move(look),
        c.focal_length,
        c.f_number,
    };
}

__device__ __host__
ray generate_ray(
    camera const& c,
    float const x,
    float const y,
    glm::vec2 const& sample,
    glm::vec2 const& sample_dof)
{
    let x_ndc =  ((x + sample.x) / c.width  * 2.0f - 1.0f);
    let y_ndc = -((y + sample.y) / c.height * 2.0f - 1.0f);

    let d = c.focal_length * glm::tan(c.fov / 2.0f);
    let a = c.width / c.height;
    let xc = d * x_ndc * a;
    let yc = d * y_ndc;
    let zc = -c.focal_length;
    glm::vec4 const pc(xc, yc, zc, 1.0f);
    let p_ = c.world_matrix * pc;

    let r = c.focal_length / (2.0f * c.f_number);
    let circle_sample = glm::vec4(
        r * concentric_sample_disk(sample_dof),
        0.0f,
        1.0f);
    let cp = c.world_matrix * circle_sample;
    let d_ = glm::normalize(p_ - cp);
    return { cp, d_ };
}
}

#endif
