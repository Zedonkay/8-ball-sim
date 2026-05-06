#pragma once

#include "sim_state.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

constexpr int DT_REDUCTION_BLOCK_SIZE = 256;
constexpr int DT_REDUCTION_SCRATCH =
    (N_MAX + DT_REDUCTION_BLOCK_SIZE - 1) / DT_REDUCTION_BLOCK_SIZE;

struct GpuRigidDie
{
    float pos_x{};
    float pos_y{};
    float pos_z{};
    float vel_x{};
    float vel_y{};
    float vel_z{};
    float omega_x{};
    float omega_y{};
    float omega_z{};
    float quat_w{};
    float quat_x{};
    float quat_y{};
    float quat_z{};
    float half_x{};
    float half_y{};
    float half_z{};
    float mass{};
    // Body-frame inverse inertia 
    float inv_ix{};
    float inv_iy{};
    float inv_iz{};
    float drag_coeff{};
    float idle_time_since_input{};
    int   n_ghost_offsets{};
    float ghost_offset_x[MAX_DIE_GHOSTS]{};
    float ghost_offset_y[MAX_DIE_GHOSTS]{};
    float ghost_offset_z[MAX_DIE_GHOSTS]{};
};

struct SimGpuRuntimeCaps
{
    float rigid_max_force{};
    float rigid_max_torque{};
    float rigid_max_speed{};
    float rigid_max_omega{};
    float die_linear_damping{};
    float die_angular_damping{};
    float die_center_spring{};
    float die_center_damping{};
    float die_control_accel_gain{};
    float die_key_velocity_gain{};
    float die_key_omega_gain{};
};

// Keyboard impulse limits 
struct ShakeUiKernels
{
    float accel_step{};
    float torque_step{};
    float accel_limit{};
    float torque_limit{};
    float decay_rate_per_sec{};
};

struct SimGpuPendingInput
{
    float        delta[6]{};
    unsigned int flags{};
    unsigned int pending{};
};

struct SimGpuBuffers
{
    float         *pos_x              = nullptr;
    float         *pos_y              = nullptr;
    float         *pos_z              = nullptr;
    float         *pos_x_sorted       = nullptr;
    float         *pos_y_sorted       = nullptr;
    float         *pos_z_sorted       = nullptr;
    float         *vel_x[2]           = {nullptr, nullptr};
    float         *vel_y[2]           = {nullptr, nullptr};
    float         *vel_z[2]           = {nullptr, nullptr};
    float         *vel_x_sorted[2]    = {nullptr, nullptr};
    float         *vel_y_sorted[2]    = {nullptr, nullptr};
    float         *vel_z_sorted[2]    = {nullptr, nullptr};
    float         *vel_xsph_x         = nullptr;
    float         *vel_xsph_y         = nullptr;
    float         *vel_xsph_z         = nullptr;
    float         *vel_xsph_x_sorted   = nullptr;
    float         *vel_xsph_y_sorted   = nullptr;
    float         *vel_xsph_z_sorted   = nullptr;
    float         *force_x            = nullptr;
    float         *force_y            = nullptr;
    float         *force_z            = nullptr;
    float         *mass               = nullptr;
    float         *mass_sorted        = nullptr;
    float         *density            = nullptr;
    float         *density_sorted     = nullptr;
    float         *pressure           = nullptr;
    float         *pressure_sorted    = nullptr;
    bool          *is_ghost           = nullptr;
    bool          *is_ghost_sorted    = nullptr;
    int           *particle_id        = nullptr;
    int           *particle_id_sorted = nullptr;
    std::uint32_t *radix_key          = nullptr;
    std::uint32_t *radix_key_alt      = nullptr;
    int           *sorted_id          = nullptr;
    int           *sorted_id_alt      = nullptr;
    int           *cell_start         = nullptr;
    int           *cell_count         = nullptr;
    int           *neighbor_list      = nullptr;
    int           *neighbor_count     = nullptr;
    int           *ghost_flags        = nullptr;
    int           *ghost_offsets      = nullptr;
    float         *dt_block_max_speed = nullptr;
    float         *dt_block_max_accel = nullptr;
    float         *die_force          = nullptr;
    float         *die_torque         = nullptr;
    float         *die_clamp_debug    = nullptr; 
    int           *d_n_total          = nullptr;
    int           *d_ping             = nullptr;
    float         *d_dt               = nullptr;
    float         *d_max_speed        = nullptr;
    float         *d_max_accel        = nullptr;
    float *d_dt_diagnostic =
        nullptr; 
    int   *d_dt_limiter                = nullptr;
    void  *cub_sort_temp               = nullptr;
    std::size_t    cub_sort_temp_bytes = 0;
    GpuRigidDie        *d_die           = nullptr;
    float              *d_control       = nullptr; // accel xyz + torque xyz
    float              *d_sim_time      = nullptr;
    int                *d_step          = nullptr;
    float              *d_prev_dt       = nullptr;
    float         *d_script_times      = nullptr;
    unsigned char *d_script_keys       = nullptr;
    int           *d_script_count      = nullptr;
    int           *d_script_cursor     = nullptr;
    float         *h_pos_x             = nullptr;
    float         *h_pos_y             = nullptr;
    float         *h_pos_z             = nullptr;
    float         *h_density           = nullptr;
    cudaStream_t   stream_compute      = nullptr;
    cudaStream_t   stream_copy         = nullptr;
    cudaEvent_t    evt_copy_ready      = nullptr;
    bool           initialized         = false;
};

int div_up(int n, int d);

bool alloc_buffers(SimGpuBuffers &b);
void free_buffers(SimGpuBuffers &b);
bool copy_to_device(void *dst, const void *src, std::size_t bytes);
bool copy_to_host(void *dst, const void *src, std::size_t bytes);
bool launch_ok();
bool sync_ok();


bool sim_cuda_upload_constant_params(const Params &params);
bool sim_cuda_upload_constant_runtime_caps(const SimGpuRuntimeCaps &caps);
bool sim_cuda_upload_constant_shake_ui(const ShakeUiKernels &k);

bool launch_init_gpu_particle_ids(int n_fluid, SimGpuBuffers &buffers);
bool launch_inject_ghost_particles(SimState &s, SimGpuBuffers &buffers);
bool launch_gather_canonical_frame_layout(int n_fluid, int n_total,
                                          SimGpuBuffers &buffers);
bool launch_density_pressure_forces(SimState &s, SimGpuBuffers &buffers);
bool launch_density_pressure_forces_reuse_neighbors(SimState &s,
                                                    SimGpuBuffers &buffers);
bool launch_density_pressure_forces_fused(SimState &s, SimGpuBuffers &buffers);
bool launch_die_coupling(SimState &s, SimGpuBuffers &buffers);
bool launch_dt_reduction(SimState &s, SimGpuBuffers &buffers);
bool launch_reduce_dt_and_finalize(SimState &s, SimGpuBuffers &buffers);
bool launch_integrate(SimState &s, SimGpuBuffers &buffers);
bool launch_boundary(SimState &s, SimGpuBuffers &buffers);
bool launch_toggle_ping(SimGpuBuffers &buffers);
bool launch_integrate_rigid_die(SimGpuBuffers             &buffers,
                                const SimGpuPendingInput &pending_input);
