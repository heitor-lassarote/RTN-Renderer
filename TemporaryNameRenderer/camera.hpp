#pragma once
#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <crt/host_defines.h>
#include <glm/vec3.hpp>

namespace renderer_temporary_name
{
struct camera
{
    glm::vec3 const position;

    __device__ __host__
    explicit camera(glm::vec3 const& position)
        : position(position)
    {
    }
};
}

#endif
