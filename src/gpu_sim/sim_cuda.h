#pragma once

#include "sim_state.h"

struct RigidDie;

struct ShakeUiHostParams
{
    float accel_step{};
    float torque_step{};
    float accel_limit{};
    float torque_limit{};
    float decay_rate_per_sec{};
};

struct SimGpuInputDelta
{
    float        delta[6]{};
    unsigned int flags{};
};

struct SimGpuParityDebug
{
    float dt{};
    float dt_advective{};
    float dt_acoustic{};
    float dt_force{};
    float dt_viscous{};
    float max_speed{};
    float max_accel{};
    int   dt_limiter{};
    float die_force_pre[3]{};
    float die_force_post[3]{};
    float die_torque_pre[3]{};
    float die_torque_post[3]{};
};

constexpr unsigned SIM_GPU_SHAKE_CLEAR_CONTROL = 1u;

bool sim_gpu_available();

bool sim_gpu_ensure_initialized(SimState &s);
void sim_gpu_shutdown(SimState &s);

bool sim_gpu_upload_die(SimState &s, const RigidDie &die);

bool sim_gpu_upload_shake_ui(const ShakeUiHostParams &p);

bool sim_gpu_upload_scripted_keys(SimState            &s,
                                  int                  n,
                                  const float         *times,
                                  const unsigned char *keys);

// One step
bool sim_gpu_advance(SimState &s, const SimGpuInputDelta *input_delta);

bool sim_gpu_pull_frame_for_write(SimState &s,
                                  RigidDie &die,
                                  int      *out_step,
                                  float    *out_t);

bool sim_gpu_pull_parity_debug(SimState &s, SimGpuParityDebug *out_debug);
