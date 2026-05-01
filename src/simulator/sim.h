#pragma once

#include "rigid_diagnostics.h"
#include "rigid_body.h"
#include "sim_state.h"

enum class DtLimiter
{
    Advective,
    Acoustic,
    Force,
    Viscous,
};

struct DtDiagnostics
{
    float     dt              = 0.0f;
    float     dt_advective    = 0.0f;
    float     dt_acoustic     = 0.0f;
    float     dt_force        = 0.0f;
    float     dt_viscous      = 0.0f;
    float     max_fluid_speed = 0.0f;
    float     max_fluid_accel = 0.0f;
    DtLimiter limiter         = DtLimiter::Advective;
};

float step_sim(SimState      &s,
               RigidDie      &die,
               const Vec3    &external_accel,
               const Vec3    &external_torque,
               DtDiagnostics *dt_diagnostic = nullptr,
               RigidClampDiagnostics *rigid_clamp_diagnostic = nullptr);
