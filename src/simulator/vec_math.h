#pragma once

#include <cmath>

struct Vec3
{
    float x{};
    float y{};
    float z{};

    Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3 &o) const
    {
        return {x + o.x, y + o.y, z + o.z};
    }
    Vec3 operator-(const Vec3 &o) const
    {
        return {x - o.x, y - o.y, z - o.z};
    }
    Vec3 operator*(float s) const
    {
        return {x * s, y * s, z * s};
    }
    Vec3 operator/(float s) const
    {
        return {x / s, y / s, z / s};
    }
    Vec3 operator-() const
    {
        return {-x, -y, -z};
    }

    Vec3 &operator+=(const Vec3 &o)
    {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }
    Vec3 &operator-=(const Vec3 &o)
    {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }
    Vec3 &operator*=(float s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    float dot(const Vec3 &o) const
    {
        return x * o.x + y * o.y + z * o.z;
    }
    Vec3 cross(const Vec3 &o) const
    {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    float length_sq() const
    {
        return x * x + y * y + z * z;
    }
    float length() const
    {
        return std::sqrt(length_sq());
    }

    Vec3 normalized() const
    {
        float len = length();
        if(len < 1e-12f)
            return {0.f, 0.f, 0.f};
        return *this / len;
    }
};

inline Vec3 operator*(float s, const Vec3 &v)
{
    return v * s;
}
