#pragma once

#include "vec_math.h"

#include <cmath>

struct die_dimensions
{
    float dist{};
    Vec3  normal{};
};

inline die_dimensions sdf_box(const Vec3 &p, const Vec3 &h)
{
    const Vec3 d = {
        std::fabs(p.x) - h.x,
        std::fabs(p.y) - h.y,
        std::fabs(p.z) - h.z,
    };

    const Vec3 d_clamped = {
        std::fmax(d.x, 0.0f),
        std::fmax(d.y, 0.0f),
        std::fmax(d.z, 0.0f),
    };

    const float outside_dist = d_clamped.length();
    const float inside_dist =
        std::fmin(std::fmax(d.x, std::fmax(d.y, d.z)), 0.0f);
    const float dist = outside_dist + inside_dist;

    Vec3 normal{};
    if(dist > 0.0f)
    {
        normal = d_clamped.normalized();
    }
    else
    {
        if(d.x > d.y && d.x > d.z)
            normal = {(p.x >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f};
        else if(d.y > d.x && d.y > d.z)
            normal = {0.0f, (p.y >= 0.0f) ? 1.0f : -1.0f, 0.0f};
        else
            normal = {0.0f, 0.0f, (p.z >= 0.0f) ? 1.0f : -1.0f};
    }

    return {dist, normal};
}
