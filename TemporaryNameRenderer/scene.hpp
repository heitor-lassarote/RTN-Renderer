#pragma once
#ifndef SCENE_HPP
#define SCENE_HPP

#include <utility>
#include <vector>

#include <crt/host_defines.h>

#include "intersection.hpp"
#include "ray.hpp"
#include "shape.hpp"
#include <functional>

//struct scene
//{
//    std::vector<shape> const shapes;
//
//    __device__
//    scene(std::vector<shape> shapes)
//        : shapes(std::move(shapes))
//    {
//    }
//};
//
//__device__
//static intersection intersection_impl(
//    scene const& s,
//    ray const& r,
//    intersection const& i,
//    size_t const index)
//{
//    if (index == s.shapes.size())
//    {
//        return i;
//    }
//    else
//    {
//        shape const object = s.shapes[index];
//        intersection const temp = intersects(object, r);
//        float const distance = temp.distance;
//        if (temp.hit && distance < i.distance)
//        {
//            return intersection_impl(
//                s,
//                r,
//                { distance, object, true },
//                index + 1);
//        }
//        else
//        {
//            return intersection_impl(s, r, i, index + 1);
//        }
//    }
//}
//
//template<>
//__device__
//intersection intersects(scene const& s, ray const& r)
//{
//    return intersection_impl(s, r, {}, 0);
//}

#endif
