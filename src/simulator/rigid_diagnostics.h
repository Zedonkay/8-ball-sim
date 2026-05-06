#pragma once

#include "vec_math.h"

struct RigidClampDiagnostics
{
    Vec3  force_pre{};
    Vec3  force_post{};
    Vec3  torque_pre{};
    Vec3  torque_post{};
    float force_pre_norm{};
    float force_post_norm{};
    float torque_pre_norm{};
    float torque_post_norm{};
};
