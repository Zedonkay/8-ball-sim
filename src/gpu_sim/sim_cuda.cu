#include "constants.h"
#include "rigid_body.h"
#include "sim_cuda.h"
#include "sim_cuda_internal.h"

#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>

namespace
{
bool upload_fluid_core_to_device(const SimState &s, SimGpuBuffers &buffers)
{
    const int n_fluid = s.n_fluid;
    const int v_read  = s.ping;
    return copy_to_device(buffers.pos_x, s.pos_x, sizeof(float) * n_fluid) &&
           copy_to_device(buffers.pos_y, s.pos_y, sizeof(float) * n_fluid) &&
           copy_to_device(buffers.pos_z, s.pos_z, sizeof(float) * n_fluid) &&
           copy_to_device(buffers.vel_x[v_read],
                          s.vel_x[v_read],
                          sizeof(float) * n_fluid) &&
           copy_to_device(buffers.vel_y[v_read],
                          s.vel_y[v_read],
                          sizeof(float) * n_fluid) &&
           copy_to_device(buffers.vel_z[v_read],
                          s.vel_z[v_read],
                          sizeof(float) * n_fluid) &&
           copy_to_device(buffers.vel_x[s.pong],
                          s.vel_x[s.pong],
                          sizeof(float) * n_fluid) &&
           copy_to_device(buffers.vel_y[s.pong],
                          s.vel_y[s.pong],
                          sizeof(float) * n_fluid) &&
           copy_to_device(buffers.vel_z[s.pong],
                          s.vel_z[s.pong],
                          sizeof(float) * n_fluid) &&
           copy_to_device(buffers.mass, s.mass, sizeof(float) * n_fluid) &&
           copy_to_device(buffers.is_ghost, s.is_ghost, sizeof(bool) * n_fluid);
}

bool zero_die_accumulators(SimGpuBuffers &buffers)
{
    return cudaMemset(buffers.die_force, 0, sizeof(float) * 3) == cudaSuccess &&
           cudaMemset(buffers.die_torque, 0, sizeof(float) * 3) == cudaSuccess;
}

GpuRigidDie make_gpu_die(const SimState &s, const RigidDie &die)
{
    GpuRigidDie g{};
    g.pos_x      = die.pos.x;
    g.pos_y      = die.pos.y;
    g.pos_z      = die.pos.z;
    g.vel_x      = die.vel.x;
    g.vel_y      = die.vel.y;
    g.vel_z      = die.vel.z;
    g.omega_x    = die.omega.x;
    g.omega_y    = die.omega.y;
    g.omega_z    = die.omega.z;
    g.quat_w     = die.orient.w;
    g.quat_x     = die.orient.x;
    g.quat_y     = die.orient.y;
    g.quat_z     = die.orient.z;
    g.half_x     = die.half_extents.x;
    g.half_y     = die.half_extents.y;
    g.half_z     = die.half_extents.z;
    g.mass       = die.mass;
    g.inv_ix     = die.inv_inertia_body.m[0][0];
    g.inv_iy     = die.inv_inertia_body.m[1][1];
    g.inv_iz     = die.inv_inertia_body.m[2][2];
    g.drag_coeff = s.params.drag_coeff;
    g.idle_time_since_input = die.idle_time_since_input;
    g.n_ghost_offsets =
        die.n_ghost_offsets < MAX_DIE_GHOSTS ? die.n_ghost_offsets
                                             : MAX_DIE_GHOSTS;
    for(int i = 0; i < g.n_ghost_offsets; ++i)
    {
        g.ghost_offset_x[i] = die.ghost_offsets[i].x;
        g.ghost_offset_y[i] = die.ghost_offsets[i].y;
        g.ghost_offset_z[i] = die.ghost_offsets[i].z;
    }
    return g;
}

void apply_gpu_die_to_host(const GpuRigidDie &g, RigidDie &die)
{
    die.pos          = {g.pos_x, g.pos_y, g.pos_z};
    die.vel          = {g.vel_x, g.vel_y, g.vel_z};
    die.omega        = {g.omega_x, g.omega_y, g.omega_z};
    die.orient.w     = g.quat_w;
    die.orient.x     = g.quat_x;
    die.orient.y     = g.quat_y;
    die.orient.z     = g.quat_z;
    die.half_extents = {g.half_x, g.half_y, g.half_z};
    die.mass         = g.mass;
    die.idle_time_since_input = g.idle_time_since_input;
    die.n_ghost_offsets = g.n_ghost_offsets;
    for(int i = 0; i < g.n_ghost_offsets; ++i)
    {
        die.ghost_offsets[i] = {
            g.ghost_offset_x[i], g.ghost_offset_y[i], g.ghost_offset_z[i]};
    }
}
}

bool sim_gpu_available()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool sim_gpu_ensure_initialized(SimState &s)
{
    auto *buffers = reinterpret_cast<SimGpuBuffers *>(s.gpu_runtime);
    if(buffers && buffers->initialized)
        return true;

    int         gpu_count = 0;
    cudaError_t count_err = cudaGetDeviceCount(&gpu_count);
    if(count_err != cudaSuccess || gpu_count <= 0)
    {
        std::fprintf(stderr,
                     "cuda device unavailable (count=%d): %s\n",
                     gpu_count,
                     cudaGetErrorString(count_err));
        return false;
    }

    auto *fresh = new SimGpuBuffers();
    if(!alloc_buffers(*fresh))
    {
        std::fprintf(stderr,
                     "gpu buffer allocation failed during initialization\n");
        free_buffers(*fresh);
        delete fresh;
        return false;
    }
    if(!upload_fluid_core_to_device(s, *fresh))
    {
        std::fprintf(
            stderr,
            "gpu initial host->device copy failed during initialization\n");
        free_buffers(*fresh);
        delete fresh;
        return false;
    }

    if(cudaMemcpy(fresh->d_n_total,
                  &s.n_total,
                  sizeof(int),
                  cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::fprintf(stderr, "gpu d_n_total init failed\n");
        free_buffers(*fresh);
        delete fresh;
        return false;
    }

    if(!launch_init_gpu_particle_ids(s.n_fluid, *fresh))
    {
        std::fprintf(stderr, "gpu particle_id init failed\n");
        free_buffers(*fresh);
        delete fresh;
        return false;
    }

    if(cudaMemcpy(
           fresh->d_ping, &s.ping, sizeof(int), cudaMemcpyHostToDevice) !=
       cudaSuccess)
    {
        std::fprintf(stderr, "gpu d_ping init failed\n");
        free_buffers(*fresh);
        delete fresh;
        return false;
    }

    if(!sim_cuda_upload_constant_params(s.params))
    {
        std::fprintf(stderr, "cudaMemcpyToSymbol c_gpu_params failed\n");
        free_buffers(*fresh);
        delete fresh;
        return false;
    }

    SimGpuRuntimeCaps caps{};
    caps.rigid_max_force     = RIGID_MAX_FORCE_RUNTIME;
    caps.rigid_max_torque    = RIGID_MAX_TORQUE_RUNTIME;
    caps.rigid_max_speed     = RIGID_MAX_SPEED_RUNTIME;
    caps.rigid_max_omega     = RIGID_MAX_OMEGA_RUNTIME;
    caps.die_linear_damping  = DIE_LINEAR_DAMPING_RUNTIME;
    caps.die_angular_damping = DIE_ANGULAR_DAMPING_RUNTIME;
    caps.die_center_spring   = DIE_CENTER_SPRING_RUNTIME;
    caps.die_center_damping    = DIE_CENTER_DAMPING_RUNTIME;
    caps.die_control_accel_gain = DIE_CONTROL_ACCEL_GAIN_RUNTIME;
    caps.die_key_velocity_gain = DIE_KEY_VELOCITY_GAIN_RUNTIME;
    caps.die_key_omega_gain    = DIE_KEY_OMEGA_GAIN_RUNTIME;
    if(!sim_cuda_upload_constant_runtime_caps(caps))
    {
        std::fprintf(stderr, "cudaMemcpyToSymbol c_runtime_caps failed\n");
        free_buffers(*fresh);
        delete fresh;
        return false;
    }

    s.gpu_runtime = fresh;
    return true;
}

void sim_gpu_shutdown(SimState &s)
{
    auto *buffers = reinterpret_cast<SimGpuBuffers *>(s.gpu_runtime);
    if(!buffers)
        return;
    free_buffers(*buffers);
    delete buffers;
    s.gpu_runtime = nullptr;
}

bool sim_gpu_upload_die(SimState &s, const RigidDie &die)
{
    auto *buffers = reinterpret_cast<SimGpuBuffers *>(s.gpu_runtime);
    if(!buffers || !buffers->initialized)
        return false;
    const GpuRigidDie g = make_gpu_die(s, die);
    return cudaMemcpy(buffers->d_die,
                      &g,
                      sizeof(GpuRigidDie),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

bool sim_gpu_upload_shake_ui(const ShakeUiHostParams &p)
{
    ShakeUiKernels k{};
    k.accel_step         = p.accel_step;
    k.torque_step        = p.torque_step;
    k.accel_limit        = p.accel_limit;
    k.torque_limit       = p.torque_limit;
    k.decay_rate_per_sec = p.decay_rate_per_sec;
    return sim_cuda_upload_constant_shake_ui(k);
}

bool sim_gpu_upload_scripted_keys(SimState            &s,
                                  int                  n,
                                  const float         *times,
                                  const unsigned char *keys)
{
    auto *b = reinterpret_cast<SimGpuBuffers *>(s.gpu_runtime);
    if(!b || !b->initialized)
        return false;

    cudaFree(b->d_script_times);
    cudaFree(b->d_script_keys);
    b->d_script_times = nullptr;
    b->d_script_keys  = nullptr;

    if(n <= 0 || !times || !keys)
    {
        const int z = 0;
        if(cudaMemcpy(
               b->d_script_count, &z, sizeof(int), cudaMemcpyHostToDevice) !=
           cudaSuccess)
            return false;
        const int zc = 0;
        return cudaMemcpy(b->d_script_cursor,
                          &zc,
                          sizeof(int),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    }

    float         *dt = nullptr;
    unsigned char *dk = nullptr;
    if(cudaMalloc(reinterpret_cast<void **>(&dt), sizeof(float) * n) !=
           cudaSuccess ||
       cudaMalloc(reinterpret_cast<void **>(&dk), n) != cudaSuccess)
    {
        cudaFree(dt);
        cudaFree(dk);
        return false;
    }
    if(cudaMemcpy(dt, times, sizeof(float) * n, cudaMemcpyHostToDevice) !=
           cudaSuccess ||
       cudaMemcpy(dk, keys, n, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(dt);
        cudaFree(dk);
        return false;
    }

    b->d_script_times = dt;
    b->d_script_keys  = dk;
    if(cudaMemcpy(b->d_script_count, &n, sizeof(int), cudaMemcpyHostToDevice) !=
       cudaSuccess)
        return false;
    const int zc = 0;
    return cudaMemcpy(
               b->d_script_cursor, &zc, sizeof(int), cudaMemcpyHostToDevice) ==
           cudaSuccess;
}

bool sim_gpu_advance(SimState &s, const SimGpuInputDelta *input_delta)
{
    static int step_counter = 0;
    auto *buffers = reinterpret_cast<SimGpuBuffers *>(s.gpu_runtime);
    if(!buffers || !buffers->initialized)
        return false;

    SimGpuPendingInput pending{};
    if(input_delta)
    {
        for(int i = 0; i < 6; ++i)
            pending.delta[i] = input_delta->delta[i];
        pending.flags   = input_delta->flags;
        pending.pending = 1u;
    }

    if(!launch_inject_ghost_particles(s, *buffers))
        return false;
    const int rebuild_every = SIM_NEIGHBOR_REBUILD_EVERY < 1 ? 1 : SIM_NEIGHBOR_REBUILD_EVERY;
    const bool do_rebuild = (step_counter % rebuild_every) == 0;
    if(SIM_GPU_FUSED_FORCES)
    {
        if(!launch_density_pressure_forces_fused(s, *buffers))
            return false;
    }
    else if(do_rebuild)
    {
        if(!launch_density_pressure_forces(s, *buffers))
            return false;
    }
    else
    {
        if(!launch_density_pressure_forces_reuse_neighbors(s, *buffers))
            return false;
    }

    if(s.n_fluid > 0)
    {
        if(!zero_die_accumulators(*buffers))
            return false;
        if(!launch_die_coupling(s, *buffers))
            return false;
        if(!launch_dt_reduction(s, *buffers))
            return false;
        if(!launch_reduce_dt_and_finalize(s, *buffers))
            return false;
    }
    else
    {
        const float dt_fallback = DT_MAX;
        if(cudaMemcpy(buffers->d_dt,
                      &dt_fallback,
                      sizeof(float),
                      cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
    }

    if(!launch_integrate(s, *buffers))
        return false;
    if(!launch_toggle_ping(*buffers))
        return false;
    if(!launch_boundary(s, *buffers))
        return false;
    if(!launch_integrate_rigid_die(*buffers, pending))
        return false;
    if(SIM_GPU_STRICT_SYNC)
    {
        if(!sync_ok())
            return false;
    }
    ++step_counter;
    return true;
}

bool sim_gpu_pull_frame_for_write(SimState &s,
                                  RigidDie &die,
                                  int      *out_step,
                                  float    *out_t)
{
    auto *buffers = reinterpret_cast<SimGpuBuffers *>(s.gpu_runtime);
    if(!buffers || !buffers->initialized || !out_step || !out_t)
        return false;

    int n_total = 0;
    if(cudaMemcpy(
           &n_total, buffers->d_n_total, sizeof(int), cudaMemcpyDeviceToHost) !=
       cudaSuccess)
        return false;
    s.n_total         = n_total;
    const int n_fluid = s.n_fluid;
    if(n_fluid < 0 || n_total < n_fluid || n_total > N_MAX)
        return false;

    float *src_x = buffers->pos_x;
    float *src_y = buffers->pos_y;
    float *src_z = buffers->pos_z;
    float *src_density = buffers->density;
    if(n_fluid > 0)
    {
        if(!launch_gather_canonical_frame_layout(n_fluid, n_total, *buffers))
            return false;
        if(cudaDeviceSynchronize() != cudaSuccess)
            return false;
        src_x       = buffers->pos_x_sorted;
        src_y       = buffers->pos_y_sorted;
        src_z       = buffers->pos_z_sorted;
        src_density = buffers->density_sorted;
    }

    if(cudaMemcpyAsync(buffers->h_pos_x,
                       src_x,
                       sizeof(float) * n_total,
                       cudaMemcpyDeviceToHost,
                       buffers->stream_copy) != cudaSuccess ||
       cudaMemcpyAsync(buffers->h_pos_y,
                       src_y,
                       sizeof(float) * n_total,
                       cudaMemcpyDeviceToHost,
                       buffers->stream_copy) != cudaSuccess ||
       cudaMemcpyAsync(buffers->h_pos_z,
                       src_z,
                       sizeof(float) * n_total,
                       cudaMemcpyDeviceToHost,
                       buffers->stream_copy) != cudaSuccess ||
       cudaMemcpyAsync(buffers->h_density,
                       src_density,
                       sizeof(float) * n_fluid,
                       cudaMemcpyDeviceToHost,
                       buffers->stream_copy) != cudaSuccess)
    {
        return false;
    }
    if(cudaEventRecord(buffers->evt_copy_ready, buffers->stream_copy) !=
       cudaSuccess)
    {
        return false;
    }
    if(cudaEventSynchronize(buffers->evt_copy_ready) != cudaSuccess)
        return false;

    std::memcpy(s.pos_x, buffers->h_pos_x, sizeof(float) * n_total);
    std::memcpy(s.pos_y, buffers->h_pos_y, sizeof(float) * n_total);
    std::memcpy(s.pos_z, buffers->h_pos_z, sizeof(float) * n_total);
    std::memcpy(s.density, buffers->h_density, sizeof(float) * n_fluid);

    GpuRigidDie g{};
    if(cudaMemcpy(
           &g, buffers->d_die, sizeof(GpuRigidDie), cudaMemcpyDeviceToHost) !=
       cudaSuccess)
        return false;
    apply_gpu_die_to_host(g, die);

    int   step_v = 0;
    float t_v    = 0.0f;
    if(cudaMemcpy(
           &step_v, buffers->d_step, sizeof(int), cudaMemcpyDeviceToHost) !=
           cudaSuccess ||
       cudaMemcpy(
           &t_v, buffers->d_sim_time, sizeof(float), cudaMemcpyDeviceToHost) !=
           cudaSuccess)
        return false;
    *out_step = step_v;
    *out_t    = t_v;
    return true;
}

bool sim_gpu_pull_parity_debug(SimState &s, SimGpuParityDebug *out_debug)
{
    auto *buffers = reinterpret_cast<SimGpuBuffers *>(s.gpu_runtime);
    if(!buffers || !buffers->initialized || !out_debug)
        return false;

    float dt_diagnostic[7] = {};
    int   limiter    = 0;
    float clamp_diag[12] = {};
    if(cudaMemcpy(dt_diagnostic,
                  buffers->d_dt_diagnostic,
                  sizeof(dt_diagnostic),
                  cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(&limiter,
                  buffers->d_dt_limiter,
                  sizeof(int),
                  cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(clamp_diag,
                  buffers->die_clamp_debug,
                  sizeof(clamp_diag),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return false;
    }

    out_debug->dt         = dt_diagnostic[0];
    out_debug->dt_advective = dt_diagnostic[1];
    out_debug->dt_acoustic  = dt_diagnostic[2];
    out_debug->dt_force     = dt_diagnostic[3];
    out_debug->dt_viscous   = dt_diagnostic[4];
    out_debug->max_speed    = dt_diagnostic[5];
    out_debug->max_accel    = dt_diagnostic[6];
    out_debug->dt_limiter   = limiter;

    for(int i = 0; i < 3; ++i)
    {
        out_debug->die_force_pre[i]  = clamp_diag[i];
        out_debug->die_torque_pre[i] = clamp_diag[3 + i];
        out_debug->die_force_post[i] = clamp_diag[6 + i];
        out_debug->die_torque_post[i] = clamp_diag[9 + i];
    }
    return true;
}
