#pragma once

#include "constants.h"
#include "quat_math.h"
#include "rigid_diagnostics.h"
#include "vec_math.h"

#include <algorithm>
#include <cstdio>

struct RigidDie
{
    Vec3 pos{};
    Vec3 vel{};
    Vec3 omega{};
    Quat orient{Quat::identity()};

    float mass{DIE_MASS};
    Vec3  half_extents{DIE_HALF, DIE_HALF, DIE_HALF};
    Mat3  inertia_body{};
    Mat3  inv_inertia_body{};

    Vec3 force_accum{};
    Vec3 torque_accum{};
    float idle_time_since_input{};

    Vec3 ghost_offsets[MAX_DIE_GHOSTS]{};
    int  n_ghost_offsets{};
};

inline bool is_finite_vec3(const Vec3 &v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline Vec3 clamp_vec3_length(const Vec3 &v, float max_len)
{
    const float len = v.length();
    if(len <= max_len || len < 1e-8f)
        return v;
    return v * (max_len / len);
}

inline void sample_box_surface_offsets(RigidDie &die,
                                       float     spacing = PARTICLE_R)
{
    const float sx = std::fmax(0.0f, die.half_extents.x - DIE_GHOST_SURFACE_INSET);
    const float sy = std::fmax(0.0f, die.half_extents.y - DIE_GHOST_SURFACE_INSET);
    const float sz = std::fmax(0.0f, die.half_extents.z - DIE_GHOST_SURFACE_INSET);
    const float step = std::fmax(0.5f * spacing, 1e-4f);

    die.n_ghost_offsets = 0;

    auto push = [&](const Vec3 &v)
    {
        if(die.n_ghost_offsets >= MAX_DIE_GHOSTS)
            return;
        die.ghost_offsets[die.n_ghost_offsets++] = v;
    };

    for(float y = -sy; y <= sy + 0.5f * step; y += step)
    {
        for(float z = -sz; z <= sz + 0.5f * step; z += step)
        {
            push({sx, std::fmin(y, sy), std::fmin(z, sz)});
            push({-sx, std::fmin(y, sy), std::fmin(z, sz)});
        }
    }
    for(float x = -sx; x <= sx + 0.5f * step; x += step)
    {
        for(float z = -sz; z <= sz + 0.5f * step; z += step)
        {
            push({std::fmin(x, sx), sy, std::fmin(z, sz)});
            push({std::fmin(x, sx), -sy, std::fmin(z, sz)});
        }
    }
    for(float x = -sx; x <= sx + 0.5f * step; x += step)
    {
        for(float y = -sy; y <= sy + 0.5f * step; y += step)
        {
            push({std::fmin(x, sx), std::fmin(y, sy), sz});
            push({std::fmin(x, sx), std::fmin(y, sy), -sz});
        }
    }
}

inline RigidDie init_rigid_die()
{
    RigidDie die{};
    die.pos          = {0.0f, 0.0f, 0.0f};
    die.vel          = {0.0f, 0.0f, 0.0f};
    die.omega        = {0.0f, 0.0f, 0.0f};
    die.orient       = Quat::identity();
    die.mass         = DIE_MASS;
    die.half_extents = {DIE_HALF, DIE_HALF, DIE_HALF};

    const float i_diag = die.mass * DIE_SIDE * DIE_SIDE / 6.0f;
    die.inertia_body   = Mat3::diagonal(i_diag, i_diag, i_diag);
    die.inv_inertia_body =
        Mat3::diagonal(1.0f / i_diag, 1.0f / i_diag, 1.0f / i_diag);

    die.force_accum  = {0.0f, 0.0f, 0.0f};
    die.torque_accum = {0.0f, 0.0f, 0.0f};
    sample_box_surface_offsets(die);
    return die;
}

inline void integrate_rigid_body(RigidDie   &die,
                                 float       dt,
                                 const Vec3 &external_accel,
                                 const Vec3 &external_torque,
                                 RigidClampDiagnostics *rigid_clamp_diagnostic = nullptr)
{
    die.force_accum += die.mass * external_accel;
    die.torque_accum += external_torque;
    die.force_accum +=
        (-DIE_CENTER_SPRING_RUNTIME) * die.pos -
        DIE_CENTER_DAMPING_RUNTIME * die.vel;
    const bool quiet_control =
        external_accel.length_sq() <
            DIE_IDLE_ACCEL_CONTROL_SLEEP * DIE_IDLE_ACCEL_CONTROL_SLEEP &&
        external_torque.length_sq() <
            DIE_IDLE_TORQUE_CONTROL_SLEEP * DIE_IDLE_TORQUE_CONTROL_SLEEP;
    if(quiet_control)
        die.idle_time_since_input += dt;
    else
        die.idle_time_since_input = 0.0f;
    const bool idle_control =
        quiet_control && die.idle_time_since_input >= DIE_IDLE_GRACE_TIME;
    if(idle_control)
    {
        if(die.force_accum.length_sq() <
           DIE_IDLE_FORCE_SLEEP * DIE_IDLE_FORCE_SLEEP)
            die.force_accum = {0.0f, 0.0f, 0.0f};
        if(die.torque_accum.length_sq() <
           DIE_IDLE_TORQUE_SLEEP * DIE_IDLE_TORQUE_SLEEP)
            die.torque_accum = {0.0f, 0.0f, 0.0f};
    }

    const Vec3 force_pre_vec  = die.force_accum;
    const Vec3 torque_pre_vec = die.torque_accum;
    const float force_pre_clamp  = die.force_accum.length();
    const float torque_pre_clamp = die.torque_accum.length();
    die.force_accum =
        clamp_vec3_length(die.force_accum, RIGID_MAX_FORCE_RUNTIME);
    die.torque_accum =
        clamp_vec3_length(die.torque_accum, RIGID_MAX_TORQUE_RUNTIME);
    if(rigid_clamp_diagnostic)
    {
        rigid_clamp_diagnostic->force_pre        = force_pre_vec;
        rigid_clamp_diagnostic->torque_pre       = torque_pre_vec;
        rigid_clamp_diagnostic->force_post       = die.force_accum;
        rigid_clamp_diagnostic->torque_post      = die.torque_accum;
        rigid_clamp_diagnostic->force_pre_norm   = force_pre_clamp;
        rigid_clamp_diagnostic->force_post_norm  = die.force_accum.length();
        rigid_clamp_diagnostic->torque_pre_norm  = torque_pre_clamp;
        rigid_clamp_diagnostic->torque_post_norm = die.torque_accum.length();
    }

    static int die_clamp_log_remaining = 40;
    if(!SIM_NO_OUTPUT && die_clamp_log_remaining > 0)
    {
        std::printf("[die_clamp] |F| pre/post=%.4f/%.4f N (cap=%.3f)  "
                    "|tau| pre/post=%.4f/%.4f N·m (cap=%.3f)\n",
                    force_pre_clamp,
                    die.force_accum.length(),
                    RIGID_MAX_FORCE_RUNTIME,
                    torque_pre_clamp,
                    die.torque_accum.length(),
                    RIGID_MAX_TORQUE_RUNTIME);
        --die_clamp_log_remaining;
    }

    // calculate buyoncy force (try to keep the die near the center)
    Vec3 buoyancy_force = NEUTRAL_BUOYANCY_MASS * (-GRAVITY);
    Vec3 total_force    = die.force_accum + die.mass * GRAVITY + buoyancy_force;
    Vec3 accel          = total_force / std::fmax(die.mass, MIN_RHO) +
                 DIE_CONTROL_ACCEL_GAIN_RUNTIME * external_accel;
    die.vel += dt * accel;
    die.vel *= std::exp(-DIE_LINEAR_DAMPING_RUNTIME * dt);
    if(idle_control)
    {
        die.vel *= std::exp(-DIE_IDLE_LINEAR_DAMPING * dt);
        if(die.vel.length_sq() < DIE_IDLE_SPEED_SLEEP * DIE_IDLE_SPEED_SLEEP)
            die.vel = {0.0f, 0.0f, 0.0f};
    }
    die.vel = clamp_vec3_length(die.vel, RIGID_MAX_SPEED_RUNTIME);
    die.pos += dt * die.vel;

    Mat3 r           = Mat3::from_quat(die.orient);
    Mat3 i_inv_world = r * die.inv_inertia_body * r.transposed();
    Vec3 alpha       = i_inv_world * die.torque_accum;
    die.omega += dt * alpha;
    die.omega *= std::exp(-DIE_ANGULAR_DAMPING_RUNTIME * dt);
    if(idle_control)
    {
        die.omega *= std::exp(-DIE_IDLE_ANGULAR_DAMPING * dt);
        if(die.omega.length_sq() < DIE_IDLE_OMEGA_SLEEP * DIE_IDLE_OMEGA_SLEEP)
            die.omega = {0.0f, 0.0f, 0.0f};
    }
    die.omega = clamp_vec3_length(die.omega, RIGID_MAX_OMEGA_RUNTIME);

    Quat omega_q{0.0f, die.omega.x, die.omega.y, die.omega.z};
    die.orient += (die.orient * omega_q) * (0.5f * dt);
    die.orient = die.orient.normalized();

    if(!is_finite_vec3(die.pos) || !is_finite_vec3(die.vel) ||
       !is_finite_vec3(die.omega))
    {
        die.pos    = {0.0f, 0.0f, -SPHERE_R * 0.3f};
        die.vel    = {0.0f, 0.0f, 0.0f};
        die.omega  = {0.0f, 0.0f, 0.0f};
        die.orient = Quat::identity();
    }

    die.force_accum  = {0.0f, 0.0f, 0.0f};
    die.torque_accum = {0.0f, 0.0f, 0.0f};
}

inline void clamp_die_inside_sphere(RigidDie &die)
{
    const float bounding_radius =
        std::sqrt(die.half_extents.x * die.half_extents.x +
                  die.half_extents.y * die.half_extents.y +
                  die.half_extents.z * die.half_extents.z);
    const float limit =
        std::fmax(0.0f, SPHERE_R - bounding_radius - DIE_WALL_CLEARANCE);
    const float r2    = die.pos.length_sq();
    if(r2 <= limit * limit)
        return;

    const float r = std::sqrt(r2);
    Vec3        n = (r > 1e-8f) ? (die.pos / r) : Vec3{0.0f, 0.0f, 1.0f};
    die.pos       = limit * n;

    float vn = die.vel.dot(n);
    if(vn > 0.0f)
        die.vel -= vn * n;
}
