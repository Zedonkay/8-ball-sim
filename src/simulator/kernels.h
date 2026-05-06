#pragma once


#include "constants.h"
#include "vec_math.h"

inline float W_poly6(float r_sq)
{
    const float h_sq = H * H;
    if(r_sq >= h_sq)
        return 0.f;
    const float q = h_sq - r_sq;
    return ALPHA_POLY6 * q * q * q;
}

inline Vec3 gradW_spiky(Vec3 r_vec, float r)
{
    if(r <= 0.f || r >= H)
        return {0.f, 0.f, 0.f};
    const float q     = H - r;
    const float coeff = -ALPHA_SPIKY * q * q / r;
    return r_vec * coeff;
}

inline float lapW_viscosity(float r)
{
    if(r >= H)
        return 0.f;
    return ALPHA_VISC * (H - r);
}
