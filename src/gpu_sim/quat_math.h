#pragma once

#include "vec_math.h"

#include <cmath>

struct Quat
{
    float w{1.0f};
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};

    static constexpr Quat identity()
    {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }

    Quat operator*(const Quat &q) const
    {
        return {
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w,
        };
    }

    Quat operator*(float s) const
    {
        return {w * s, x * s, y * s, z * s};
    }

    Quat operator+(const Quat &q) const
    {
        return {w + q.w, x + q.x, y + q.y, z + q.z};
    }

    Quat &operator+=(const Quat &q)
    {
        w += q.w;
        x += q.x;
        y += q.y;
        z += q.z;
        return *this;
    }

    Quat conjugate() const
    {
        return {w, -x, -y, -z};
    }

    Quat normalized() const
    {
        const float n2 = w * w + x * x + y * y + z * z;
        if(!std::isfinite(n2) || n2 < 1e-12f)
            return identity();
        const float inv_n = 1.0f / std::sqrt(n2);
        return {w * inv_n, x * inv_n, y * inv_n, z * inv_n};
    }

    Vec3 rotate(const Vec3 &v) const
    {
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        const float xy = x * y;
        const float xz = x * z;
        const float yz = y * z;
        const float wx = w * x;
        const float wy = w * y;
        const float wz = w * z;

        return {
            v.x * (1.0f - 2.0f * (yy + zz)) + v.y * (2.0f * (xy - wz)) +
                v.z * (2.0f * (xz + wy)),
            v.x * (2.0f * (xy + wz)) + v.y * (1.0f - 2.0f * (xx + zz)) +
                v.z * (2.0f * (yz - wx)),
            v.x * (2.0f * (xz - wy)) + v.y * (2.0f * (yz + wx)) +
                v.z * (1.0f - 2.0f * (xx + yy)),
        };
    }

    Vec3 inverse_rotate(const Vec3 &v) const
    {
        return conjugate().rotate(v);
    }
};

inline Quat operator*(float s, const Quat &q)
{
    return q * s;
}

struct Mat3
{
    float m[3][3]{};

    static Mat3 diagonal(float a, float b, float c)
    {
        Mat3 out{};
        out.m[0][0] = a;
        out.m[1][1] = b;
        out.m[2][2] = c;
        return out;
    }

    static Mat3 from_quat(const Quat &q_in)
    {
        const Quat  q  = q_in.normalized();
        const float xx = q.x * q.x;
        const float yy = q.y * q.y;
        const float zz = q.z * q.z;
        const float xy = q.x * q.y;
        const float xz = q.x * q.z;
        const float yz = q.y * q.z;
        const float wx = q.w * q.x;
        const float wy = q.w * q.y;
        const float wz = q.w * q.z;

        Mat3 out{};
        out.m[0][0] = 1.0f - 2.0f * (yy + zz);
        out.m[0][1] = 2.0f * (xy - wz);
        out.m[0][2] = 2.0f * (xz + wy);
        out.m[1][0] = 2.0f * (xy + wz);
        out.m[1][1] = 1.0f - 2.0f * (xx + zz);
        out.m[1][2] = 2.0f * (yz - wx);
        out.m[2][0] = 2.0f * (xz - wy);
        out.m[2][1] = 2.0f * (yz + wx);
        out.m[2][2] = 1.0f - 2.0f * (xx + yy);
        return out;
    }

    Mat3 operator*(const Mat3 &o) const
    {
        Mat3 out{};
        for(int r = 0; r < 3; ++r)
        {
            for(int c = 0; c < 3; ++c)
            {
                float acc = 0.0f;
                for(int k = 0; k < 3; ++k)
                {
                    acc += m[r][k] * o.m[k][c];
                }
                out.m[r][c] = acc;
            }
        }
        return out;
    }

    Vec3 operator*(const Vec3 &v) const
    {
        return {
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        };
    }

    Mat3 transposed() const
    {
        Mat3 out{};
        for(int r = 0; r < 3; ++r)
        {
            for(int c = 0; c < 3; ++c)
            {
                out.m[r][c] = m[c][r];
            }
        }
        return out;
    }
};
