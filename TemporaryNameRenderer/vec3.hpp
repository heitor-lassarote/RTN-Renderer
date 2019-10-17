#ifndef VEC3H
#define VEC3H

#include "device_launch_parameters.h"

#include <iostream>

class vec3
{
public:
    union
    {
        struct
        {
            float x;
            float y;
            float z;
        };

        struct
        {
            float r;
            float g;
            float b;
        };
    };

    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0, float e1, float e2) { x = e0; y = e1; z = e2; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-x, -y, -z); }
    __host__ __device__ inline float operator[](int i) const
    {
        return *reinterpret_cast<float const*>(this + i * sizeof(float));
    }
    __host__ __device__ inline float& operator[](int i)
    {
        return *reinterpret_cast<float*>(this + i * sizeof(float));
    }

    __host__ __device__ inline vec3& operator+=(const vec3 &v2);
    __host__ __device__ inline vec3& operator-=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const vec3 &v2);
    __host__ __device__ inline vec3& operator/=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    __host__ __device__ inline float length() const
    {
        return sqrt(squared_length());
    }
    __host__ __device__ inline float squared_length() const
    {
        return x * x + y * y + z * z;
    }

    __host__ __device__ inline void make_unit_vector();
};

inline std::istream& operator>>(std::istream &is, vec3 &t)
{
    is >> t.x >> t.y >> t.z;
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t)
{
    os << t.x << " " << t.y << " " << t.z;
    return os;
}

__host__ __device__ inline void vec3::make_unit_vector()
{
    float k = 1.0 / sqrt(x*x + y*y + z*z);
    x *= k; y *= k; z *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
{
    return vec3(t*v.x, t*v.y, t*v.z);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
    return vec3(v.x/t, v.y/t, v.z/t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t)
{
    return vec3(t*v.x, t*v.y, t*v.z);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2)
{
    return v1.x *v2.x + v1.y *v2.y  + v1.z *v2.z;
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3(
        v1.y*v2.z - v1.z*v2.y,
        -(v1.x*v2.z - v1.z*v2.x),
         v1.x*v2.y - v1.y*v2.x);
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v)
{
    x  += v.x;
    y  += v.y;
    z  += v.z;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v)
{
    x  *= v.x;
    y  *= v.y;
    z  *= v.z;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v)
{
    x  /= v.x;
    y  /= v.y;
    z  /= v.z;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v)
{
    x  -= v.x;
    y  -= v.y;
    z  -= v.z;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t)
{
    x  *= t;
    y  *= t;
    z  *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t)
{
    float k = 1.0/t;

    x  *= k;
    y  *= k;
    z  *= k;
    return *this;
}

__host__ __device__ inline vec3 normalize(vec3 const& v)
{
    return v / v.length();
}

#endif
