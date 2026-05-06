#pragma once

#include <cstddef>
#include <termios.h>
#include <utility>
#include <vector>

#include "vec_math.h"

struct ShakeInputState
{
    Vec3  accel                             = {0.0f, 0.0f, 0.0f};
    Vec3  torque                            = {0.0f, 0.0f, 0.0f};
    float accel_step                        = 15.0f;
    float accel_limit                       = 30.0f;
    float torque_step                       = 0.02f;
    float torque_limit                      = 0.25f;
    float min_accel_step                    = 0.1f;
    float max_accel_step                    = 50.0f;
    float min_torque_step                   = 0.001f;
    float max_torque_step                   = 0.1f;
    bool  reduce_opposite_axis_cancellation = false;
    float opposite_axis_scale               = 0.35f;
    bool  output_enabled                    = true;
    bool  announce_requires_verbose         = false;
    float decay_rate_per_sec                = 10.1f;

    bool pending_clear_control = false;

    void apply_axis_impulse(float &axis, float delta);
    void clamp_accel();
    void print_input_state() const;
    void handle_key(unsigned char c, bool announce = true);
    void decay_inputs(float dt);
};

struct ScriptedKeyEvents
{
    std::vector<std::pair<float, unsigned char>> events;
    std::size_t                                  next = 0;

    bool load(const char *path,
              bool        verbose,
              bool        output_enabled,
              bool        warn_malformed_without_verbose = false);
    void apply_through_time(float t, ShakeInputState &shake);
};

struct KeyboardControl
{
    bool    enabled = false;
    bool    verbose = false;
    int     fd      = -1;
    termios old_term{};
    int     old_flags = 0;

    ShakeInputState *shake = nullptr;

    bool init(ShakeInputState *state, bool verbose_logs);
    void shutdown();
    ~KeyboardControl();
    void poll();
};

void enable_high_agitation_profile(ShakeInputState &shake);
void enable_stable_live_profile(ShakeInputState &shake);
void print_keyboard_controls();
