#pragma once
#ifndef SAMPLE_SCENES_HPP
#define SAMPLE_SCENES_HPP

#include <glm/glm.hpp>

#include "let.hpp"
#include "triangle.hpp"

namespace rtn
{
void compass_scene(triangle* triangles, size_t& count)
{
    triangles[count++] = triangle
    {
        bsdf{ bsdf_type::none, glm::vec3(0.0f, 1.0f, 0.0f) },
        vertex
        {
            glm::vec3(-0.5f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.5f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.0f, 1.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.5f, 1.0f)
        },
    };

    triangles[count++] = triangle
    {
        bsdf{ bsdf_type::none, glm::vec3(0.0f, 1.0f, 0.0f) },
        vertex
        {
            glm::vec3(0.5f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(-0.5f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.0f, -1.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.5f, 1.0f)
        },
    };

    triangles[count++] = triangle
    {
        bsdf{ bsdf_type::none, glm::vec3(0.0f, 1.0f, 0.0f) },
        vertex
        {
            glm::vec3(0.0f, 0.5f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.0f, -0.5f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(1.0f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.5f, 1.0f)
        },
    };

    triangles[count++] = triangle
    {
        bsdf{ bsdf_type::none, glm::vec3(0.0f, 1.0f, 0.0f) },
        vertex
        {
            glm::vec3(0.0f, -0.5f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.0f, 0.5f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(-1.0f, 0.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.5f, 1.0f)
        },
    };

    triangles[count++] = triangle
    {
        bsdf{ bsdf_type::light, glm::vec3(1.0f, 1.0f, 1.0f) },
        vertex
        {
            glm::vec3(0.0f, -0.5f, -0.5f),
            glm::vec3(0.0f, 0.0f, -1.0f),
            glm::vec2(0.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(0.0f, 0.5f, -0.5f),
            glm::vec3(0.0f, 0.0f, -1.0f),
            glm::vec2(1.0f, 0.0f)
        },
        vertex
        {
            glm::vec3(-1.0f, 0.0f, -0.5f),
            glm::vec3(0.0f, 0.0f, -1.0f),
            glm::vec2(0.5f, 1.0f)
        },
    };
}

void compass_scene(triangle* triangles)
{
    size_t count = 0;
    compass_scene(triangles, count);
}

void simple_scene(triangle* triangles, size_t& count)
{
    glm::vec3 const red(1.0f, 0.0f, 0.0f);
    glm::vec3 const blue(0.0f, 0.0f, 1.0f);
    glm::vec3 const white(1.0f, 1.0f, 1.0f);
    glm::vec3 const yellow(1.0f, 1.0f, 0.0f);

    float const bottom = -1.0f;
    float const top = 1.0f;
    float const left = -1.0f;
    float const right = 1.0f;
    float const front = 1.0f;
    float const back = -1.0f;

    // building
    // bottom
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, white },
        vertex
        {
            glm::vec3(left, bottom, front),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, bottom, front),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, bottom, back),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(1.0f, 1.0f),
        },
    };
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, white },
        vertex
        {
            glm::vec3(right, bottom, back),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(1.0f, 1.0f),
        },
        vertex
        {
            glm::vec3(left, bottom, back),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(0.0f, 1.0f),
        },
        vertex
        {
            glm::vec3(left, bottom, front),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };

    // top
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, white },
        vertex
        {
            glm::vec3(left, top, front),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, top, front),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, top, back),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(1.0f, 1.0f),
        },
    };
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, white },
        vertex
        {
            glm::vec3(right, top, back),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(1.0f, 1.0f),
        },
        vertex
        {
            glm::vec3(left, top, back),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(0.0f, 1.0f),
        },
        vertex
        {
            glm::vec3(left, top, front),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };

    // back
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, white },
        vertex
        {
            glm::vec3(left, bottom, back),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, bottom, back),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, top, back),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, white },
        vertex
        {
            glm::vec3(right, top, back),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(left, top, back),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(left, bottom, back),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };

    // left
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, red },
        vertex
        {
            glm::vec3(left, bottom, back),
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(left, bottom, front),
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(left, top, front),
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, red },
        vertex
        {
            glm::vec3(left, top, front),
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(left, top, back),
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(left, bottom, back),
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };

    // right
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, blue },
        vertex
        {
            glm::vec3(right, bottom, front),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, bottom, back),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, top, back),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::diffuse, blue },
        vertex
        {
            glm::vec3(right, top, back),
            glm::vec3(-0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, top, front),
            glm::vec3(-0.0f, 0.0f, 1.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(right, bottom, front),
            glm::vec3(-0.0f, 0.0f, 1.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };

    // light
    let factor = 0.5f;
    let l_left = left * factor;
    let l_right = right * factor;
    let l_top = top - 0.01f;
    let l_front = front * factor;
    let l_back = back * factor;

    let light_color = glm::vec3(2.0f);

    triangles[count++] = triangle
    {
        bsdf { bsdf_type::light, light_color },
        vertex
        {
            glm::vec3(l_left, l_top, l_front),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(l_right, l_top, l_front),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(l_right, l_top, l_back),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };
    triangles[count++] = triangle
    {
        bsdf { bsdf_type::light, light_color },
        vertex
        {
            glm::vec3(l_right, l_top, l_back),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(l_left, l_top, l_back),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            glm::vec3(l_left, l_top, l_front),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };
}

void simple_scene(triangle* triangles)
{
    size_t count = 0;
    simple_scene(triangles, count);
}

void quad(
    triangle* triangles,
    glm::vec4 const& bottom_left,
    glm::vec4 const& bottom_right,
    glm::vec4 const& top_right,
    glm::vec4 const& top_left,
    bsdf const& bsdf,
    glm::mat4 const& transform,
    size_t& count)
{
    triangles[count++] = triangle
    {
        bsdf,
        vertex
        {
            transform * bottom_left,
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
        vertex
        {
            transform * bottom_right,
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(1.0f, 0.0f),
        },
        vertex
        {
            transform * top_right,
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(1.0f, 1.0f),
        },
    };
    triangles[count++] = triangle
    {
        bsdf,
        vertex
        {
            transform * top_right,
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(1.0f, 1.0f),
        },
        vertex
        {
            transform * top_left,
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(0.0f, 1.0f),
        },
        vertex
        {
            transform * bottom_left,
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec2(0.0f, 0.0f),
        },
    };
}

void quad(
    triangle* triangles,
    glm::vec4 const& bottom_left,
    glm::vec4 const& bottom_right,
    glm::vec4 const& top_right,
    glm::vec4 const& top_left,
    bsdf const& bsdf,
    size_t& count)
{
    quad(
        triangles,
        bottom_left,
        bottom_right,
        top_right,
        top_left,
        bsdf,
        glm::mat4(1.0f),
        count);
}

void box(
    triangle* triangles,
    bsdf const& bsdf,
    glm::mat4 const& transform,
    size_t& count)
{
    let size = 0.5f;
    let bottom = -size;
    let top = size;
    let left = -size;
    let right = size;
    let front = size;
    let back = -size;

    quad(
        triangles,
        glm::vec4(left,  bottom, front, 1.0f),
        glm::vec4(right, bottom, front, 1.0f),
        glm::vec4(right, bottom, back,  1.0f),
        glm::vec4(left,  bottom, back,  1.0f),
        bsdf,
        transform,
        count);
    quad(
        triangles,
        glm::vec4(left,  top,    front, 1.0f),
        glm::vec4(right, top,    front, 1.0f),
        glm::vec4(right, top,    back,  1.0f),
        glm::vec4(left,  top,    back,  1.0f),
        bsdf,
        transform,
        count);
    quad(
        triangles,
        glm::vec4(left,  bottom, front, 1.0f),
        glm::vec4(right, bottom, front, 1.0f),
        glm::vec4(right, top,    front, 1.0f),
        glm::vec4(left,  top,    front, 1.0f),
        bsdf,
        transform,
        count);
    quad(
        triangles,
        glm::vec4(left,  bottom, back,  1.0f),
        glm::vec4(right, bottom, back,  1.0f),
        glm::vec4(right, top,    back,  1.0f),
        glm::vec4(left,  top,    back,  1.0f),
        bsdf,
        transform,
        count);
    quad(
        triangles,
        glm::vec4(left,  bottom, back,  1.0f),
        glm::vec4(left,  bottom, front, 1.0f),
        glm::vec4(left,  top,    front, 1.0f),
        glm::vec4(left,  top,    back,  1.0f),
        bsdf,
        transform,
        count);
    quad(
        triangles,
        glm::vec4(right, bottom, front, 1.0f),
        glm::vec4(right, bottom, back,  1.0f),
        glm::vec4(right, top,    back,  1.0f),
        glm::vec4(right, top,    front, 1.0f),
        bsdf,
        transform,
        count);
}
}

#endif
