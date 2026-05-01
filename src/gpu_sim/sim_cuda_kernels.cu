#include "constants.h"
#include "sim_cuda_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

__constant__ Params            c_gpu_params;
__constant__ SimGpuRuntimeCaps c_runtime_caps;
__constant__ ShakeUiKernels    c_shake_ui;

namespace
{
__device__ inline float d_w_poly6(float r_sq)
{
    float q = H * H - r_sq;
    if(q <= 0.0f)
        return 0.0f;
    return ALPHA_POLY6 * q * q * q;
}

__device__ inline float d_lap_w_visc(float r)
{
    if(r >= H)
        return 0.0f;
    return ALPHA_VISC * (H - r);
}

__device__ inline int d_cell_coord(float v)
{
    return static_cast<int>(floorf(v / CELL_SIZE));
}

__device__ inline int d_cell_index_biased(int ix, int iy, int iz)
{
    const int bx = max(0, min(GRID_N_AXIS - 1, ix + GRID_BIAS));
    const int by = max(0, min(GRID_N_AXIS - 1, iy + GRID_BIAS));
    const int bz = max(0, min(GRID_N_AXIS - 1, iz + GRID_BIAS));
    return (bz * GRID_N_AXIS + by) * GRID_N_AXIS + bx;
}

__device__ inline void d_grad_w_spiky(
    float dx, float dy, float dz, float r_sq, float inv_r, float &gx, float &gy, float &gz)
{
    const float r = r_sq * inv_r;
    if(r <= 0.0f || r >= H)
    {
        gx = gy = gz = 0.0f;
        return;
    }
    float q     = H - r;
    float scale = -ALPHA_SPIKY * q * q / r;
    gx          = scale * dx;
    gy          = scale * dy;
    gz          = scale * dz;
}

struct DtReduceValue
{
    float speed;
    float accel;
};

struct DtReduceMax
{
    __device__ DtReduceValue operator()(const DtReduceValue &a,
                                        const DtReduceValue &b) const
    {
        return {fmaxf(a.speed, b.speed), fmaxf(a.accel, b.accel)};
    }
};

struct DeviceVec3
{
    float x;
    float y;
    float z;
};

struct Devicedie_dimensions
{
    float      dist;
    DeviceVec3 normal;
};

__device__ inline DeviceVec3 d_sub(const DeviceVec3 &a, const DeviceVec3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ inline DeviceVec3 d_add(const DeviceVec3 &a, const DeviceVec3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ inline DeviceVec3 d_scale(float s, const DeviceVec3 &v)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ inline float d_dot(const DeviceVec3 &a, const DeviceVec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline DeviceVec3 d_cross(const DeviceVec3 &a, const DeviceVec3 &b)
{
    return {
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__device__ inline float d_length_sq(const DeviceVec3 &v)
{
    return d_dot(v, v);
}

__device__ inline DeviceVec3 d_normalized(const DeviceVec3 &v)
{
    float len_sq = d_length_sq(v);
    if(len_sq <= 1e-24f)
        return {0.0f, 0.0f, 0.0f};
    float inv_len = rsqrtf(len_sq);
    return d_scale(inv_len, v);
}

__device__ inline const float *
d_vel_read(const float *v_ping, const float *v_pong, const int *d_ping)
{
    return (*d_ping == PING) ? v_ping : v_pong;
}

__device__ inline float *
d_vel_write(float *v_ping, float *v_pong, const int *d_ping)
{
    return (*d_ping == PING) ? v_pong : v_ping;
}

__device__ inline float *
d_vel_active(float *v_ping, float *v_pong, const int *d_ping)
{
    return (*d_ping == PING) ? v_ping : v_pong;
}

__device__ inline DeviceVec3
d_quat_rotate(float w, float x, float y, float z, const DeviceVec3 &v)
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

    return {v.x * (1.0f - 2.0f * (yy + zz)) + v.y * (2.0f * (xy - wz)) +
                v.z * (2.0f * (xz + wy)),
            v.x * (2.0f * (xy + wz)) + v.y * (1.0f - 2.0f * (xx + zz)) +
                v.z * (2.0f * (yz - wx)),
            v.x * (2.0f * (xz - wy)) + v.y * (2.0f * (yz + wx)) +
                v.z * (1.0f - 2.0f * (xx + yy))};
}

__device__ inline DeviceVec3
d_quat_inverse_rotate(float w, float x, float y, float z, const DeviceVec3 &v)
{
    return d_quat_rotate(w, -x, -y, -z, v);
}

__device__ inline Devicedie_dimensions
d_sdf_box(const DeviceVec3 &p, float hx, float hy, float hz)
{
    const DeviceVec3 d = {fabsf(p.x) - hx, fabsf(p.y) - hy, fabsf(p.z) - hz};
    const DeviceVec3 d_clamped = {
        fmaxf(d.x, 0.0f), fmaxf(d.y, 0.0f), fmaxf(d.z, 0.0f)};

    const float outside_dist = sqrtf(d_length_sq(d_clamped));
    const float inside_dist  = fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.0f);
    const float dist         = outside_dist + inside_dist;

    DeviceVec3 normal{};
    if(dist > 0.0f)
    {
        normal = d_normalized(d_clamped);
    }
    else if(d.x > d.y && d.x > d.z)
    {
        normal = {(p.x >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f};
    }
    else if(d.y > d.x && d.y > d.z)
    {
        normal = {0.0f, (p.y >= 0.0f) ? 1.0f : -1.0f, 0.0f};
    }
    else
    {
        normal = {0.0f, 0.0f, (p.z >= 0.0f) ? 1.0f : -1.0f};
    }

    return {dist, normal};
}

__global__ void kernel_hash_particles(const int     *d_n_total,
                                      const float   *pos_x,
                                      const float   *pos_y,
                                      const float   *pos_z,
                                      std::uint32_t *radix_key,
                                      int           *sorted_id)
{
    int       i  = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt = *d_n_total;
    if(i >= nt)
    {
        if(i < N_MAX)
        {
            radix_key[i] = static_cast<std::uint32_t>(TABLE_SIZE);
            sorted_id[i] = i;
        }
        return;
    }

    int ix       = d_cell_coord(pos_x[i]);
    int iy       = d_cell_coord(pos_y[i]);
    int iz       = d_cell_coord(pos_z[i]);
    radix_key[i] = static_cast<std::uint32_t>(d_cell_index_biased(ix, iy, iz));
    sorted_id[i] = i;
}

__global__ void kernel_reorder_particle_arrays(
    const int     *d_n_total,
    const int     *sorted_id,
    const float   *in_pos_x,
    const float   *in_pos_y,
    const float   *in_pos_z,
    const float   *in_vel_x_ping,
    const float   *in_vel_y_ping,
    const float   *in_vel_z_ping,
    const float   *in_vel_x_pong,
    const float   *in_vel_y_pong,
    const float   *in_vel_z_pong,
    const float   *in_vel_xsph_x,
    const float   *in_vel_xsph_y,
    const float   *in_vel_xsph_z,
    const float   *in_mass,
    const float   *in_density,
    const float   *in_pressure,
    const bool    *in_is_ghost,
    const int     *in_particle_id,
    float         *out_pos_x,
    float         *out_pos_y,
    float         *out_pos_z,
    float         *out_vel_x_ping,
    float         *out_vel_y_ping,
    float         *out_vel_z_ping,
    float         *out_vel_x_pong,
    float         *out_vel_y_pong,
    float         *out_vel_z_pong,
    float         *out_vel_xsph_x,
    float         *out_vel_xsph_y,
    float         *out_vel_xsph_z,
    float         *out_mass,
    float         *out_density,
    float         *out_pressure,
    bool          *out_is_ghost,
    int           *out_particle_id)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt = *d_n_total;
    if(i >= nt)
        return;
    const int src = sorted_id[i];
    out_pos_x[i] = in_pos_x[src];
    out_pos_y[i] = in_pos_y[src];
    out_pos_z[i] = in_pos_z[src];
    out_vel_x_ping[i] = in_vel_x_ping[src];
    out_vel_y_ping[i] = in_vel_y_ping[src];
    out_vel_z_ping[i] = in_vel_z_ping[src];
    out_vel_x_pong[i] = in_vel_x_pong[src];
    out_vel_y_pong[i] = in_vel_y_pong[src];
    out_vel_z_pong[i] = in_vel_z_pong[src];
    out_vel_xsph_x[i] = in_vel_xsph_x[src];
    out_vel_xsph_y[i] = in_vel_xsph_y[src];
    out_vel_xsph_z[i] = in_vel_xsph_z[src];
    out_mass[i] = in_mass[src];
    out_density[i] = in_density[src];
    out_pressure[i] = in_pressure[src];
    out_is_ghost[i] = in_is_ghost[src];
    out_particle_id[i] = in_particle_id[src];
}

__global__ void kernel_reset_sorted_id(const int *d_n_total, int *sorted_id)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt = *d_n_total;
    if(i < nt)
        sorted_id[i] = i;
}



__global__ void kernel_fill_cell_ranges(const int           *d_n_total,
                                        const std::uint32_t *radix_key,
                                        int                 *cell_start,
                                        int                 *cell_end)
{
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt = *d_n_total;
    if(rank >= nt)
        return;
    const int key = static_cast<int>(radix_key[rank]);
    if(rank == 0 || radix_key[rank - 1] != radix_key[rank])
    {
        cell_start[key] = rank;
    }
    if(rank == nt - 1 || radix_key[rank + 1] != radix_key[rank])
    {
        cell_end[key] = rank + 1;
    }
}

__global__ void kernel_init_particle_ids(int n_fluid, int *particle_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_fluid)
        particle_id[i] = i;
    else if(i < N_MAX)
        particle_id[i] = -1;
}

__global__ void kernel_fill_fluid_compact_stencil(const int   *d_n_total,
                                                  int          n_fluid,
                                                  const bool  *is_ghost,
                                                  int         *stencil_out)
{
    int       i      = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_prev = (*d_n_total > 0) ? *d_n_total : n_fluid;
    if(i >= N_MAX)
        return;
    stencil_out[i] = (i < n_prev && !is_ghost[i]) ? 1 : 0;
}

__global__ void kernel_compact_real_fluid_to_prefix(
    const int   *d_n_total,
    int          n_fluid,
    const int   *compact_dest,
    const float *in_pos_x,
    const float *in_pos_y,
    const float *in_pos_z,
    const float *in_vel_x_ping,
    const float *in_vel_y_ping,
    const float *in_vel_z_ping,
    const float *in_vel_x_pong,
    const float *in_vel_y_pong,
    const float *in_vel_z_pong,
    const float *in_vel_xsph_x,
    const float *in_vel_xsph_y,
    const float *in_vel_xsph_z,
    const float *in_mass,
    const float *in_density,
    const float *in_pressure,
    const bool  *in_is_ghost,
    const int   *in_particle_id,
    float       *out_pos_x,
    float       *out_pos_y,
    float       *out_pos_z,
    float       *out_vel_x_ping,
    float       *out_vel_y_ping,
    float       *out_vel_z_ping,
    float       *out_vel_x_pong,
    float       *out_vel_y_pong,
    float       *out_vel_z_pong,
    float       *out_vel_xsph_x,
    float       *out_vel_xsph_y,
    float       *out_vel_xsph_z,
    float       *out_mass,
    float       *out_density,
    float       *out_pressure,
    bool        *out_is_ghost,
    int         *out_particle_id)
{
    int       i      = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_prev = (*d_n_total > 0) ? *d_n_total : n_fluid;
    if(i >= n_prev || in_is_ghost[i])
        return;
    const int k = compact_dest[i];
    if(k < 0 || k >= N_MAX)
        return;
    out_pos_x[k]       = in_pos_x[i];
    out_pos_y[k]       = in_pos_y[i];
    out_pos_z[k]       = in_pos_z[i];
    out_vel_x_ping[k]  = in_vel_x_ping[i];
    out_vel_y_ping[k]  = in_vel_y_ping[i];
    out_vel_z_ping[k]  = in_vel_z_ping[i];
    out_vel_x_pong[k]  = in_vel_x_pong[i];
    out_vel_y_pong[k]  = in_vel_y_pong[i];
    out_vel_z_pong[k]  = in_vel_z_pong[i];
    out_vel_xsph_x[k]  = in_vel_xsph_x[i];
    out_vel_xsph_y[k]  = in_vel_xsph_y[i];
    out_vel_xsph_z[k]  = in_vel_xsph_z[i];
    out_mass[k]        = in_mass[i];
    out_density[k]     = in_density[i];
    out_pressure[k]    = in_pressure[i];
    out_is_ghost[k]    = false;
    out_particle_id[k] = in_particle_id[i];
}

__global__ void kernel_mark_ghost_particles(int          n_fluid,
                                            const float *pos_x,
                                            const float *pos_y,
                                            const float *pos_z,
                                            int         *ghost_flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_fluid)
        return;

    float px           = pos_x[i];
    float py           = pos_y[i];
    float pz           = pos_z[i];
    float r            = sqrtf(px * px + py * py + pz * pz);
    float dist_to_wall = SPHERE_R - r;
    ghost_flags[i]     = (dist_to_wall < H) ? 1 : 0;
}

__global__ void kernel_scatter_ghost_particles(int        n_fluid,
                                               float     *pos_x,
                                               float     *pos_y,
                                               float     *pos_z,
                                               float     *vel_x_ping,
                                               float     *vel_y_ping,
                                               float     *vel_z_ping,
                                               float     *vel_x_pong,
                                               float     *vel_y_pong,
                                               float     *vel_z_pong,
                                               float     *vel_xsph_x,
                                               float     *vel_xsph_y,
                                               float     *vel_xsph_z,
                                               float     *mass,
                                               float     *density,
                                               float     *pressure,
                                               bool      *is_ghost,
                                               int       *particle_id,
                                               const int *ghost_flags,
                                               const int         *ghost_offsets,
                                               const GpuRigidDie *gpu_die,
                                               int               *d_n_total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int last        = n_fluid - 1;
    const int wall_count  = ghost_offsets[last] + ghost_flags[last];
    const int wall_capped = min(wall_count, N_MAX - n_fluid);
    const int base        = n_fluid + wall_capped;
    const int die_count =
        max(0, min(gpu_die->n_ghost_offsets, MAX_DIE_GHOSTS));
    const int capped = min(die_count, max(0, N_MAX - base));

    if(i == 0)
    {
        *d_n_total = base + capped;
    }

    if(i < capped)
    {
        const DeviceVec3 local = {
            gpu_die->ghost_offset_x[i],
            gpu_die->ghost_offset_y[i],
            gpu_die->ghost_offset_z[i]};
        const DeviceVec3 r_world =
            d_quat_rotate(gpu_die->quat_w,
                          gpu_die->quat_x,
                          gpu_die->quat_y,
                          gpu_die->quat_z,
                          local);
        const DeviceVec3 die_pos = {gpu_die->pos_x,
                                    gpu_die->pos_y,
                                    gpu_die->pos_z};
        const DeviceVec3 p_world = d_add(die_pos, r_world);
        const DeviceVec3 die_vel = {gpu_die->vel_x,
                                    gpu_die->vel_y,
                                    gpu_die->vel_z};
        const DeviceVec3 omega = {gpu_die->omega_x,
                                  gpu_die->omega_y,
                                  gpu_die->omega_z};
        const DeviceVec3 v_surf = d_add(die_vel, d_cross(omega, r_world));
        const int        dg     = base + i;

        pos_x[dg] = p_world.x;
        pos_y[dg] = p_world.y;
        pos_z[dg] = p_world.z;

        vel_x_ping[dg] = v_surf.x;
        vel_y_ping[dg] = v_surf.y;
        vel_z_ping[dg] = v_surf.z;
        vel_x_pong[dg] = v_surf.x;
        vel_y_pong[dg] = v_surf.y;
        vel_z_pong[dg] = v_surf.z;

        vel_xsph_x[dg] = 0.0f;
        vel_xsph_y[dg] = 0.0f;
        vel_xsph_z[dg] = 0.0f;

        mass[dg]        = mass[0] * DIE_GHOST_MASS_SCALE;
        density[dg]     = c_gpu_params.rest_density;
        pressure[dg]    = 0.0f;
        is_ghost[dg]    = true;
        particle_id[dg] = -1;
    }

    if(i >= n_fluid || ghost_flags[i] == 0)
        return;

    int g = n_fluid + ghost_offsets[i];
    if(g >= N_MAX)
        return;

    float px           = pos_x[i];
    float py           = pos_y[i];
    float pz           = pos_z[i];
    float r            = sqrtf(px * px + py * py + pz * pz);
    float reciprocal_r = (r > 1e-12f) ? (1.0f / r) : 0.0f;
    float nx           = px * reciprocal_r;
    float ny           = py * reciprocal_r;
    float nz           = pz * reciprocal_r;
    if(r <= 1e-12f)
    {
        nx = 0.0f;
        ny = 0.0f;
        nz = 1.0f;
    }

    float mirror_r = 2.0f * SPHERE_R - r;
    pos_x[g]       = nx * mirror_r;
    pos_y[g]       = ny * mirror_r;
    pos_z[g]       = nz * mirror_r;

    vel_x_ping[g] = 0.0f;
    vel_y_ping[g] = 0.0f;
    vel_z_ping[g] = 0.0f;
    vel_x_pong[g] = 0.0f;
    vel_y_pong[g] = 0.0f;
    vel_z_pong[g] = 0.0f;

    vel_xsph_x[g] = 0.0f;
    vel_xsph_y[g] = 0.0f;
    vel_xsph_z[g] = 0.0f;

    density[g]  = c_gpu_params.rest_density;
    pressure[g] = 0.0f;
    is_ghost[g] = true;
    particle_id[g] = -1;
    mass[g]     = mass[i];
}

__global__ void kernel_scatter_fluid_fields_by_id(int          n_total,
                                                    int          n_fluid_cap,
                                                    const float *pos_x,
                                                    const float *pos_y,
                                                    const float *pos_z,
                                                    const float *density,
                                                    const bool  *is_ghost,
                                                    const int   *particle_id,
                                                    float       *out_pos_x,
                                                    float       *out_pos_y,
                                                    float       *out_pos_z,
                                                    float       *out_density)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_total || is_ghost[i])
        return;
    const int pid = particle_id[i];
    if(pid < 0 || pid >= n_fluid_cap)
        return;
    out_pos_x[pid]    = pos_x[i];
    out_pos_y[pid]    = pos_y[i];
    out_pos_z[pid]    = pos_z[i];
    out_density[pid] = density[i];
}

__global__ void kernel_fill_ghost_gather_stencil(int          n_total,
                                                 const bool  *is_ghost,
                                                 int         *stencil_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_total)
        return;
    stencil_out[i] = is_ghost[i] ? 1 : 0;
}

__global__ void kernel_scatter_ghost_positions_to_tail(
    int          n_total,
    int          n_fluid,
    const int   *ghost_dest,
    const float *pos_x,
    const float *pos_y,
    const float *pos_z,
    const bool  *is_ghost,
    float       *out_pos_x,
    float       *out_pos_y,
    float       *out_pos_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_total || !is_ghost[i])
        return;
    const int slot = n_fluid + ghost_dest[i];
    if(slot < 0 || slot >= N_MAX)
        return;
    out_pos_x[slot] = pos_x[i];
    out_pos_y[slot] = pos_y[i];
    out_pos_z[slot] = pos_z[i];
}

__global__ void kernel_build_neighbor_lists(const int   *d_n_total,
                                            const float *pos_x,
                                            const float *pos_y,
                                            const float *pos_z,
                                            const int   *sorted_id,
                                            const std::uint32_t *radix_key,
                                            const int   *cell_start,
                                            const int   *cell_count,
                                            const bool  *is_ghost,
                                            int         *neighbor_list,
                                            int         *neighbor_count)
{
    int       i       = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_total = *d_n_total;
    if(i >= n_total)
        return;
    if(is_ghost[i])
    {
        neighbor_count[i] = 0;
        return;
    }

    const float h_sq  = H * H;
    int         count = 0;
    int         ix    = d_cell_coord(pos_x[i]);
    int         iy    = d_cell_coord(pos_y[i]);
    int         iz    = d_cell_coord(pos_z[i]);

    for(int dx = -1; dx <= 1; ++dx)
    {
        for(int dy = -1; dy <= 1; ++dy)
        {
            for(int dz = -1; dz <= 1; ++dz)
            {
                const int target_ix = ix + dx;
                const int target_iy = iy + dy;
                const int target_iz = iz + dz;
                int       key = d_cell_index_biased(target_ix, target_iy, target_iz);
                int       start        = cell_start[key];
                int       end          = cell_count[key];
                if(start < 0 || end <= start)
                    continue;
                for(int off = start; off < end; ++off)
                {
                    int j = sorted_id[off];
                    if(j == i)
                        continue;

                    float dxp  = pos_x[i] - pos_x[j];
                    float dyp  = pos_y[i] - pos_y[j];
                    float dzp  = pos_z[i] - pos_z[j];
                    float r_sq = dxp * dxp + dyp * dyp + dzp * dzp;
                    if(r_sq < h_sq && count < MAX_NEIGHBORS)
                    {
                        neighbor_list[count * N_MAX + i] = j;
                        ++count;
                    }
                }
            }
        }
    }

    neighbor_count[i] = count;
}

__global__ void kernel_compute_density_pressure(const int   *d_n_total,
                                                const int   *neighbor_count,
                                                const int   *neighbor_list,
                                                const float *pos_x,
                                                const float *pos_y,
                                                const float *pos_z,
                                                const float *mass,
                                                const bool  *is_ghost,
                                                float       *density,
                                                float       *pressure)
{
    int       i       = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_total = *d_n_total;
    if(i >= n_total)
        return;
    if(is_ghost[i])
    {
        density[i]  = c_gpu_params.rest_density;
        pressure[i] = 0.0f;
        return;
    }

    float rho = mass[i] * d_w_poly6(0.0f);
    int   nc  = neighbor_count[i];
    for(int kk = 0; kk < nc; ++kk)
    {
        int   j    = neighbor_list[kk * N_MAX + i];
        float dx   = pos_x[i] - pos_x[j];
        float dy   = pos_y[i] - pos_y[j];
        float dz   = pos_z[i] - pos_z[j];
        float r_sq = dx * dx + dy * dy + dz * dz;
        rho += mass[j] * d_w_poly6(r_sq);
    }
    rho        = fmaxf(rho, MIN_RHO);
    density[i] = rho;

    const float rest_density = c_gpu_params.rest_density;
    const float stiffness    = c_gpu_params.stiffness;
    float       ratio        = rho / rest_density;
    float       r3           = ratio * ratio * ratio;
    float       r7           = r3 * r3 * ratio;
    float       p            = stiffness * (r7 - 1.0f);
    pressure[i]              = fmaxf(0.0f, p);
}

__global__ void kernel_compute_forces(const int   *d_n_total,
                                      const int   *neighbor_count,
                                      const int   *neighbor_list,
                                      const float *pos_x,
                                      const float *pos_y,
                                      const float *pos_z,
                                      const float *vel_x_ping,
                                      const float *vel_x_pong,
                                      const float *vel_y_ping,
                                      const float *vel_y_pong,
                                      const float *vel_z_ping,
                                      const float *vel_z_pong,
                                      const int   *d_ping,
                                      const float *mass,
                                      const float *density,
                                      const float *pressure,
                                      const bool  *is_ghost,
                                      float       *force_x,
                                      float       *force_y,
                                      float       *force_z,
                                      float       *vel_xsph_x,
                                      float       *vel_xsph_y,
                                      float       *vel_xsph_z)
{
    int       i       = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_total = *d_n_total;
    if(i >= n_total)
        return;
    if(is_ghost[i])
    {
        force_x[i]    = 0.0f;
        force_y[i]    = 0.0f;
        force_z[i]    = 0.0f;
        vel_xsph_x[i] = 0.0f;
        vel_xsph_y[i] = 0.0f;
        vel_xsph_z[i] = 0.0f;
        return;
    }

    const float *vel_x_read = d_vel_read(vel_x_ping, vel_x_pong, d_ping);
    const float *vel_y_read = d_vel_read(vel_y_ping, vel_y_pong, d_ping);
    const float *vel_z_read = d_vel_read(vel_z_ping, vel_z_pong, d_ping);

    float fp_x = 0.0f, fp_y = 0.0f, fp_z = 0.0f;
    float fv_x = 0.0f, fv_y = 0.0f, fv_z = 0.0f;
    float xs_x = 0.0f, xs_y = 0.0f, xs_z = 0.0f;

    float       vx_i      = vel_x_read[i];
    float       vy_i      = vel_y_read[i];
    float       vz_i      = vel_z_read[i];
    float       rho_i     = fmaxf(density[i], MIN_RHO);
    float       p_i       = pressure[i];
    float       m_i       = mass[i];
    const float p_i_term  = p_i / (rho_i * rho_i);
    const float h_sq      = H * H;
    const float viscosity = c_gpu_params.viscosity;

    int nc = neighbor_count[i];
    for(int kk = 0; kk < nc; ++kk)
    {
        int   j    = neighbor_list[kk * N_MAX + i];
        float dx   = pos_x[i] - pos_x[j];
        float dy   = pos_y[i] - pos_y[j];
        float dz   = pos_z[i] - pos_z[j];
        float r_sq = dx * dx + dy * dy + dz * dz;
        if(r_sq < MIN_R2 || r_sq >= h_sq)
            continue;

        float inv_r = rsqrtf(r_sq);
        float rho_j = fmaxf(density[j], MIN_RHO);
        float p_j   = pressure[j];
        float m_j   = mass[j];

        float gx, gy, gz;
        d_grad_w_spiky(dx, dy, dz, r_sq, inv_r, gx, gy, gz);
        float sym_p   = p_i_term + p_j / (rho_j * rho_j);
        float coeff_p = m_i * m_j * sym_p;
        fp_x -= coeff_p * gx;
        fp_y -= coeff_p * gy;
        fp_z -= coeff_p * gz;

        float lap_w   = d_lap_w_visc(r_sq * inv_r);
        float coeff_v = m_i * viscosity * m_j / (rho_i * rho_j) * lap_w;
        fv_x += coeff_v * (vel_x_read[j] - vx_i);
        fv_y += coeff_v * (vel_y_read[j] - vy_i);
        fv_z += coeff_v * (vel_z_read[j] - vz_i);

        const float ghost_mask = is_ghost[j] ? 0.0f : 1.0f;
        float       w_p6       = d_w_poly6(r_sq);
        float       w_coeff    = ghost_mask * (m_j / rho_j) * w_p6;
        xs_x += w_coeff * (vel_x_read[j] - vx_i);
        xs_y += w_coeff * (vel_y_read[j] - vy_i);
        xs_z += w_coeff * (vel_z_read[j] - vz_i);
    }

    force_x[i]    = fp_x + fv_x + m_i * GRAVITY.x;
    force_y[i]    = fp_y + fv_y + m_i * GRAVITY.y;
    force_z[i]    = fp_z + fv_z + m_i * GRAVITY.z;
    vel_xsph_x[i] = XSPH_EPS * xs_x;
    vel_xsph_y[i] = XSPH_EPS * xs_y;
    vel_xsph_z[i] = XSPH_EPS * xs_z;
}

__global__ void kernel_die_coupling(const int         *d_n_total,
                                    const GpuRigidDie *gpu_die,
                                    const float       *pos_x,
                                    const float       *pos_y,
                                    const float       *pos_z,
                                    const float       *vel_x_ping,
                                    const float       *vel_x_pong,
                                    const float       *vel_y_ping,
                                    const float       *vel_y_pong,
                                    const float       *vel_z_ping,
                                    const float       *vel_z_pong,
                                    const int         *d_ping,
                                    const bool        *is_ghost,
                                    const float       *density,
                                    float             *force_x,
                                    float             *force_y,
                                    float             *force_z,
                                    float             *die_force,
                                    float             *die_torque)
{
    __shared__ float block_force[3];
    __shared__ float block_torque[3];
    if(threadIdx.x < 3)
    {
        block_force[threadIdx.x] = 0.0f;
        block_torque[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int       i       = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_total = *d_n_total;
    if(i < n_total && !is_ghost[i])
    {
        const float *vel_x_read = d_vel_read(vel_x_ping, vel_x_pong, d_ping);
        const float *vel_y_read = d_vel_read(vel_y_ping, vel_y_pong, d_ping);
        const float *vel_z_read = d_vel_read(vel_z_ping, vel_z_pong, d_ping);

        GpuRigidDie      die       = *gpu_die;
        const DeviceVec3 die_pos   = {die.pos_x, die.pos_y, die.pos_z};
        const DeviceVec3 die_vel   = {die.vel_x, die.vel_y, die.vel_z};
        const DeviceVec3 die_omega = {die.omega_x, die.omega_y, die.omega_z};
        const DeviceVec3 p         = {pos_x[i], pos_y[i], pos_z[i]};
        const DeviceVec3 r_world   = d_sub(p, die_pos);
        const DeviceVec3 local_pos = d_quat_inverse_rotate(
            die.quat_w, die.quat_x, die.quat_y, die.quat_z, r_world);
        const Devicedie_dimensions sdf =
            d_sdf_box(local_pos, die.half_x, die.half_y, die.half_z);
        if(sdf.dist < DIE_CONTACT_MARGIN)
        {
            const DeviceVec3 world_normal = d_quat_rotate(
                die.quat_w, die.quat_x, die.quat_y, die.quat_z, sdf.normal);
            const float penetration = fmaxf(0.0f, DIE_CONTACT_MARGIN - sdf.dist);
            DeviceVec3  fc          = d_scale(CONTACT_K * penetration, world_normal);

            const DeviceVec3 v_surf  = d_add(die_vel, d_cross(die_omega, r_world));
            const DeviceVec3 v_fluid = {vel_x_read[i], vel_y_read[i], vel_z_read[i]};
            const DeviceVec3 v_rel   = d_sub(v_fluid, v_surf);
            const float      vn_rel  = d_dot(v_rel, world_normal);
            if(vn_rel < 0.0f)
            {
                fc = d_add(fc, d_scale(-DIE_CONTACT_DAMPING * vn_rel, world_normal));
            }

            const float speed_sq = d_length_sq(v_rel);
            DeviceVec3  fd{0.0f, 0.0f, 0.0f};
            if(speed_sq > 1e-8f)
            {
                float drag_mag =
                    0.5f * density[i] * speed_sq * die.drag_coeff * PARTICLE_AREA;
                drag_mag = fminf(drag_mag, 0.5f * c_runtime_caps.rigid_max_force);
                fd       = d_scale(drag_mag * rsqrtf(speed_sq), v_rel);
            }

            force_x[i] += fc.x - fd.x;
            force_y[i] += fc.y - fd.y;
            force_z[i] += fc.z - fd.z;

            const DeviceVec3 react  = {-fc.x + fd.x, -fc.y + fd.y, -fc.z + fd.z};
            const DeviceVec3 torque = d_cross(r_world, react);
            atomicAdd(&block_force[0], react.x);
            atomicAdd(&block_force[1], react.y);
            atomicAdd(&block_force[2], react.z);
            atomicAdd(&block_torque[0], torque.x);
            atomicAdd(&block_torque[1], torque.y);
            atomicAdd(&block_torque[2], torque.z);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        atomicAdd(&die_force[0], block_force[0]);
        atomicAdd(&die_force[1], block_force[1]);
        atomicAdd(&die_force[2], block_force[2]);
        atomicAdd(&die_torque[0], block_torque[0]);
        atomicAdd(&die_torque[1], block_torque[1]);
        atomicAdd(&die_torque[2], block_torque[2]);
    }
}

__global__ void kernel_dt_block_max(const int   *d_n_total,
                                    const float *vel_x_ping,
                                    const float *vel_x_pong,
                                    const float *vel_y_ping,
                                    const float *vel_y_pong,
                                    const float *vel_z_ping,
                                    const float *vel_z_pong,
                                    const int   *d_ping,
                                    const bool  *is_ghost,
                                    const float *force_x,
                                    const float *force_y,
                                    const float *force_z,
                                    const float *mass,
                                    float       *block_max_speed,
                                    float       *block_max_accel)
{
    using BlockReduce =
        cub::BlockReduce<DtReduceValue, DT_REDUCTION_BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const float *vel_x_read = d_vel_read(vel_x_ping, vel_x_pong, d_ping);
    const float *vel_y_read = d_vel_read(vel_y_ping, vel_y_pong, d_ping);
    const float *vel_z_read = d_vel_read(vel_z_ping, vel_z_pong, d_ping);

    int           i       = blockIdx.x * blockDim.x + threadIdx.x;
    const int     n_total = *d_n_total;
    DtReduceValue local{0.0f, 0.0f};
    if(i < n_total && !is_ghost[i])
    {
        float vx    = vel_x_read[i];
        float vy    = vel_y_read[i];
        float vz    = vel_z_read[i];
        local.speed = sqrtf(vx * vx + vy * vy + vz * vz);

        float inv_mass = 1.0f / fmaxf(mass[i], MIN_RHO);
        float ax       = force_x[i] * inv_mass;
        float ay       = force_y[i] * inv_mass;
        float az       = force_z[i] * inv_mass;
        local.accel    = sqrtf(ax * ax + ay * ay + az * az);
    }

    DtReduceValue block_value =
        BlockReduce(temp_storage).Reduce(local, DtReduceMax{});

    if(threadIdx.x == 0)
    {
        block_max_speed[blockIdx.x] = block_value.speed;
        block_max_accel[blockIdx.x] = block_value.accel;
    }
}

__global__ void kernel_reduce_dt_and_finalize(int          block_count,
                                              const float *block_max_speed,
                                              const float *block_max_accel,
                                              float       *d_max_speed,
                                              float       *d_max_accel,
                                              float       *d_dt,
                                              float       *d_dt_diagnostic,
                                              int         *d_dt_limiter)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    float ms = 0.0f;
    float ma = 0.0f;
    for(int i = 0; i < block_count; ++i)
    {
        ms = fmaxf(ms, block_max_speed[i]);
        ma = fmaxf(ma, block_max_accel[i]);
    }
    *d_max_speed = ms;
    *d_max_accel = ma;

    const float eps       = 1e-6f;
    const float max_speed = ms;
    const float max_accel = ma;
    const float rho0      = fmaxf(c_gpu_params.rest_density, MIN_RHO);
    const float nu        = c_gpu_params.viscosity / rho0;
    const float c0_from_eos =
        sqrtf(static_cast<float>(GAMMA) * c_gpu_params.stiffness / rho0);
    const float c_s = fmaxf(CFL_SOUND_SPEED_VEL_SCALE * max_speed, c0_from_eos);

    const float dt_advective = CFL_LAMBDA * H / (max_speed + eps);
    const float dt_acoustic  = CFL_LAMBDA * H / (c_s + eps);
    const float dt_force     = CFL_LAMBDA * sqrtf(H / (max_accel + eps));
    const float dt_viscous   = CFL_LAMBDA * H * H / (nu + eps);

    float dt      = dt_advective;
    int   limiter = 0; // Advective
    if(dt_acoustic < dt)
    {
        dt      = dt_acoustic;
        limiter = 1;
    }
    if(dt_force < dt)
    {
        dt      = dt_force;
        limiter = 2;
    }
    if(dt_viscous < dt)
    {
        dt      = dt_viscous;
        limiter = 3;
    }
    dt            = fmaxf(DT_MIN, fminf(dt, DT_MAX));
    *d_dt         = dt;
    *d_dt_limiter = limiter;
    if(d_dt_diagnostic)
    {
        d_dt_diagnostic[0] = dt;
        d_dt_diagnostic[1] = dt_advective;
        d_dt_diagnostic[2] = dt_acoustic;
        d_dt_diagnostic[3] = dt_force;
        d_dt_diagnostic[4] = dt_viscous;
        d_dt_diagnostic[5] = max_speed;
        d_dt_diagnostic[6] = max_accel;
    }
}

__global__ void kernel_toggle_ping(int *d_ping)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    const int p = *d_ping;
    *d_ping     = (p == PING) ? PONG : PING;
}

__device__ void d_mat3_transpose(const float A[9], float At[9])
{
    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 3; ++c)
        {
            At[r * 3 + c] = A[c * 3 + r];
        }
    }
}

__device__ void d_mat3_mul(const float A[9], const float B[9], float C[9])
{
    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 3; ++c)
        {
            float acc = 0.0f;
            for(int k = 0; k < 3; ++k)
            {
                acc += A[r * 3 + k] * B[k * 3 + c];
            }
            C[r * 3 + c] = acc;
        }
    }
}

__device__ void d_mat3_mul_diag_right(
    const float A[9], float d0, float d1, float d2, float C[9])
{
    for(int r = 0; r < 3; ++r)
    {
        C[r * 3 + 0] = A[r * 3 + 0] * d0;
        C[r * 3 + 1] = A[r * 3 + 1] * d1;
        C[r * 3 + 2] = A[r * 3 + 2] * d2;
    }
}

__device__ void
d_mat3_from_quat(float qw, float qx, float qy, float qz, float R[9])
{
    const float n2    = qw * qw + qx * qx + qy * qy + qz * qz;
    const float inv_n = rsqrtf(fmaxf(n2, 1e-12f));
    const float w     = qw * inv_n;
    const float x     = qx * inv_n;
    const float y     = qy * inv_n;
    const float z     = qz * inv_n;
    const float xx    = x * x;
    const float yy    = y * y;
    const float zz    = z * z;
    const float xy    = x * y;
    const float xz    = x * z;
    const float yz    = y * z;
    const float wx    = w * x;
    const float wy    = w * y;
    const float wz    = w * z;

    R[0] = 1.0f - 2.0f * (yy + zz);
    R[1] = 2.0f * (xy - wz);
    R[2] = 2.0f * (xz + wy);
    R[3] = 2.0f * (xy + wz);
    R[4] = 1.0f - 2.0f * (xx + zz);
    R[5] = 2.0f * (yz - wx);
    R[6] = 2.0f * (xz - wy);
    R[7] = 2.0f * (yz + wx);
    R[8] = 1.0f - 2.0f * (xx + yy);
}

__device__ void d_mat3_vec(const float M[9],
                           float       vx,
                           float       vy,
                           float       vz,
                           float      *ox,
                           float      *oy,
                           float      *oz)
{
    *ox = M[0] * vx + M[1] * vy + M[2] * vz;
    *oy = M[3] * vx + M[4] * vy + M[5] * vz;
    *oz = M[6] * vx + M[7] * vy + M[8] * vz;
}

__device__ void
d_clamp_vec3_in_place(float *x, float *y, float *z, float max_len)
{
    const float len = sqrtf(*x * *x + *y * *y + *z * *z);
    if(len <= max_len || len < 1e-8f)
        return;
    const float s = max_len / len;
    *x *= s;
    *y *= s;
    *z *= s;
}

__device__ void d_script_apply_key(unsigned char c, float *co);

__global__ void kernel_integrate_rigid_die(GpuRigidDie *die,
                                           const float *d_dt,
                                           float       *d_control,
                                           float       *d_die_force,
                                           float       *d_die_torque,
                                           float       *d_die_clamp_debug,
                                           float       *d_sim_time,
                                           int         *d_step,
                                           float       *d_prev_dt,
                                           SimGpuPendingInput pending_input,
                                           const float *d_script_times,
                                           const unsigned char *d_script_keys,
                                           const int *d_script_n_ptr,
                                           int       *d_script_cursor)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    const float dt = *d_dt;
    float       fx = d_die_force[0];
    float       fy = d_die_force[1];
    float       fz = d_die_force[2];
    float       tx = d_die_torque[0];
    float       ty = d_die_torque[1];
    float       tz = d_die_torque[2];

    float control[6];
    float kick[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for(int i = 0; i < 6; ++i)
        control[i] = d_control[i];

    const int n = *d_script_n_ptr;
    if(n > 0 && d_script_times && d_script_keys)
    {
        int         sc = *d_script_cursor;
        const float t  = *d_sim_time;
        while(sc < n && d_script_times[sc] <= t)
        {
            float before[6];
#pragma unroll
            for(int i = 0; i < 6; ++i)
                before[i] = control[i];
            d_script_apply_key(d_script_keys[sc], control);
#pragma unroll
            for(int i = 0; i < 6; ++i)
                kick[i] += control[i] - before[i];
            ++sc;
        }
        *d_script_cursor = sc;
    }

    if(pending_input.pending)
    {
        if(pending_input.flags & 1u)
        {
#pragma unroll
            for(int i = 0; i < 6; ++i)
                control[i] = 0.0f;
        }
        else
        {
#pragma unroll
            for(int i = 0; i < 6; ++i)
            {
                control[i] += pending_input.delta[i];
                kick[i] += pending_input.delta[i];
            }
        }
    }

    const float prev_dt   = *d_prev_dt;
    const float decay_mul = (prev_dt > 0.0f)
                                ? expf(-c_shake_ui.decay_rate_per_sec * prev_dt)
                                : 1.0f;
#pragma unroll
    for(int i = 0; i < 6; ++i)
    {
        control[i] *= decay_mul;
        d_control[i] = control[i];
    }
    const float control_accel_mag2 = control[0] * control[0] +
                                     control[1] * control[1] +
                                     control[2] * control[2];
    const float control_torque_mag2 = control[3] * control[3] +
                                      control[4] * control[4] +
                                      control[5] * control[5];
    const float kick_mag2 = kick[0] * kick[0] + kick[1] * kick[1] +
                            kick[2] * kick[2] + kick[3] * kick[3] +
                            kick[4] * kick[4] + kick[5] * kick[5];
    const bool quiet_control =
        control_accel_mag2 <
            DIE_IDLE_ACCEL_CONTROL_SLEEP * DIE_IDLE_ACCEL_CONTROL_SLEEP &&
        control_torque_mag2 <
            DIE_IDLE_TORQUE_CONTROL_SLEEP * DIE_IDLE_TORQUE_CONTROL_SLEEP &&
        kick_mag2 < 1e-10f;
    if(quiet_control)
        die->idle_time_since_input += dt;
    else
        die->idle_time_since_input = 0.0f;
    const bool idle_control =
        quiet_control && die->idle_time_since_input >= DIE_IDLE_GRACE_TIME;

    fx += -c_runtime_caps.die_center_spring * die->pos_x -
          c_runtime_caps.die_center_damping * die->vel_x;
    fy += -c_runtime_caps.die_center_spring * die->pos_y -
          c_runtime_caps.die_center_damping * die->vel_y;
    fz += -c_runtime_caps.die_center_spring * die->pos_z -
          c_runtime_caps.die_center_damping * die->vel_z;

    const float pre_fx = fx;
    const float pre_fy = fy;
    const float pre_fz = fz;
    const float pre_tx = tx;
    const float pre_ty = ty;
    const float pre_tz = tz;
    if(idle_control)
    {
        const float force_sleep2 = DIE_IDLE_FORCE_SLEEP * DIE_IDLE_FORCE_SLEEP;
        const float torque_sleep2 =
            DIE_IDLE_TORQUE_SLEEP * DIE_IDLE_TORQUE_SLEEP;
        if(fx * fx + fy * fy + fz * fz < force_sleep2)
            fx = fy = fz = 0.0f;
        if(tx * tx + ty * ty + tz * tz < torque_sleep2)
            tx = ty = tz = 0.0f;
    }
    d_clamp_vec3_in_place(&fx, &fy, &fz, c_runtime_caps.rigid_max_force);
    d_clamp_vec3_in_place(&tx, &ty, &tz, c_runtime_caps.rigid_max_torque);
    if(d_die_clamp_debug)
    {
        d_die_clamp_debug[0]  = pre_fx;
        d_die_clamp_debug[1]  = pre_fy;
        d_die_clamp_debug[2]  = pre_fz;
        d_die_clamp_debug[3]  = pre_tx;
        d_die_clamp_debug[4]  = pre_ty;
        d_die_clamp_debug[5]  = pre_tz;
        d_die_clamp_debug[6]  = fx;
        d_die_clamp_debug[7]  = fy;
        d_die_clamp_debug[8]  = fz;
        d_die_clamp_debug[9]  = tx;
        d_die_clamp_debug[10] = ty;
        d_die_clamp_debug[11] = tz;
    }

    const float total_fx =
        fx + die->mass * GRAVITY.x + NEUTRAL_BUOYANCY_MASS * (-GRAVITY.x);
    const float total_fy =
        fy + die->mass * GRAVITY.y + NEUTRAL_BUOYANCY_MASS * (-GRAVITY.y);
    const float total_fz =
        fz + die->mass * GRAVITY.z + NEUTRAL_BUOYANCY_MASS * (-GRAVITY.z);

    const float inv_m = 1.0f / fmaxf(die->mass, MIN_RHO);
    die->vel_x +=
        dt * (total_fx * inv_m +
              c_runtime_caps.die_control_accel_gain * control[0]);
    die->vel_y +=
        dt * (total_fy * inv_m +
              c_runtime_caps.die_control_accel_gain * control[1]);
    die->vel_z +=
        dt * (total_fz * inv_m +
              c_runtime_caps.die_control_accel_gain * control[2]);
    die->vel_x += c_runtime_caps.die_key_velocity_gain * kick[0];
    die->vel_y += c_runtime_caps.die_key_velocity_gain * kick[1];
    die->vel_z += c_runtime_caps.die_key_velocity_gain * kick[2];

    const float ld       = c_runtime_caps.die_linear_damping;
    const float damp_lin = expf(-ld * dt);
    die->vel_x *= damp_lin;
    die->vel_y *= damp_lin;
    die->vel_z *= damp_lin;
    if(idle_control)
    {
        const float idle_damp = expf(-DIE_IDLE_LINEAR_DAMPING * dt);
        die->vel_x *= idle_damp;
        die->vel_y *= idle_damp;
        die->vel_z *= idle_damp;
        if(die->vel_x * die->vel_x + die->vel_y * die->vel_y +
               die->vel_z * die->vel_z <
           DIE_IDLE_SPEED_SLEEP * DIE_IDLE_SPEED_SLEEP)
        {
            die->vel_x = die->vel_y = die->vel_z = 0.0f;
        }
    }
    d_clamp_vec3_in_place(
        &die->vel_x, &die->vel_y, &die->vel_z, c_runtime_caps.rigid_max_speed);

    die->pos_x += dt * die->vel_x;
    die->pos_y += dt * die->vel_y;
    die->pos_z += dt * die->vel_z;

    float R[9]{};
    d_mat3_from_quat(die->quat_w, die->quat_x, die->quat_y, die->quat_z, R);
    float RD[9]{};
    d_mat3_mul_diag_right(R, die->inv_ix, die->inv_iy, die->inv_iz, RD);
    float RT[9]{};
    d_mat3_transpose(R, RT);
    float Iinv[9]{};
    d_mat3_mul(RD, RT, Iinv);

    float ax{}, ay{}, az{};
    d_mat3_vec(Iinv, tx, ty, tz, &ax, &ay, &az);
    die->omega_x += dt * ax;
    die->omega_y += dt * ay;
    die->omega_z += dt * az;
    die->omega_x += c_runtime_caps.die_key_omega_gain * kick[3];
    die->omega_y += c_runtime_caps.die_key_omega_gain * kick[4];
    die->omega_z += c_runtime_caps.die_key_omega_gain * kick[5];

    const float ad       = c_runtime_caps.die_angular_damping;
    const float damp_ang = expf(-ad * dt);
    die->omega_x *= damp_ang;
    die->omega_y *= damp_ang;
    die->omega_z *= damp_ang;
    if(idle_control)
    {
        const float idle_damp = expf(-DIE_IDLE_ANGULAR_DAMPING * dt);
        die->omega_x *= idle_damp;
        die->omega_y *= idle_damp;
        die->omega_z *= idle_damp;
        if(die->omega_x * die->omega_x + die->omega_y * die->omega_y +
               die->omega_z * die->omega_z <
           DIE_IDLE_OMEGA_SLEEP * DIE_IDLE_OMEGA_SLEEP)
        {
            die->omega_x = die->omega_y = die->omega_z = 0.0f;
        }
    }
    d_clamp_vec3_in_place(&die->omega_x,
                          &die->omega_y,
                          &die->omega_z,
                          c_runtime_caps.rigid_max_omega);

    const float ox   = die->omega_x;
    const float oy   = die->omega_y;
    const float oz   = die->omega_z;
    const float qw   = die->quat_w;
    const float qx   = die->quat_x;
    const float qy   = die->quat_y;
    const float qz   = die->quat_z;
    const float half = 0.5f * dt;
    die->quat_w += half * (-qx * ox - qy * oy - qz * oz);
    die->quat_x += half * (qw * ox + qy * oz - qz * oy);
    die->quat_y += half * (qw * oy + qz * ox - qx * oz);
    die->quat_z += half * (qw * oz + qx * oy - qy * ox);

    float n2 = die->quat_w * die->quat_w + die->quat_x * die->quat_x +
               die->quat_y * die->quat_y + die->quat_z * die->quat_z;
    if(!isfinite(n2) || n2 < 1e-12f)
    {
        die->quat_w = 1.0f;
        die->quat_x = die->quat_y = die->quat_z = 0.0f;
    }
    else
    {
        const float inv = rsqrtf(n2);
        die->quat_w *= inv;
        die->quat_x *= inv;
        die->quat_y *= inv;
        die->quat_z *= inv;
    }

    if(!isfinite(die->pos_x) || !isfinite(die->pos_y) ||
       !isfinite(die->pos_z) || !isfinite(die->vel_x) ||
       !isfinite(die->vel_y) || !isfinite(die->vel_z) ||
       !isfinite(die->omega_x) || !isfinite(die->omega_y) ||
       !isfinite(die->omega_z))
    {
        die->pos_x = 0.0f;
        die->pos_y = 0.0f;
        die->pos_z = -SPHERE_R * 0.3f;
        die->vel_x = die->vel_y = die->vel_z = 0.0f;
        die->omega_x = die->omega_y = die->omega_z = 0.0f;
        die->quat_w                                = 1.0f;
        die->quat_x = die->quat_y = die->quat_z = 0.0f;
    }

    d_die_force[0] = d_die_force[1] = d_die_force[2] = 0.0f;
    d_die_torque[0] = d_die_torque[1] = d_die_torque[2] = 0.0f;

    const float hx              = die->half_x;
    const float hy              = die->half_y;
    const float hz              = die->half_z;
    const float bounding_radius = sqrtf(hx * hx + hy * hy + hz * hz);
    const float limit =
        fmaxf(0.0f, SPHERE_R - bounding_radius - DIE_WALL_CLEARANCE);
    const float px              = die->pos_x;
    const float py              = die->pos_y;
    const float pz              = die->pos_z;
    const float r2              = px * px + py * py + pz * pz;
    if(r2 > limit * limit)
    {
        const float r  = sqrtf(r2);
        float       nx = (r > 1e-8f) ? (px / r) : 0.0f;
        float       ny = (r > 1e-8f) ? (py / r) : 0.0f;
        float       nz = (r > 1e-8f) ? (pz / r) : 1.0f;
        die->pos_x     = limit * nx;
        die->pos_y     = limit * ny;
        die->pos_z     = limit * nz;

        const float vn = die->vel_x * nx + die->vel_y * ny + die->vel_z * nz;
        if(vn > 0.0f)
        {
            die->vel_x -= vn * nx;
            die->vel_y -= vn * ny;
            die->vel_z -= vn * nz;
        }
    }

    *d_sim_time += dt;
    *d_step += 1;
    *d_prev_dt = dt;
}

__global__ void kernel_integrate(const int   *d_n_total,
                                 const float *d_dt,
                                 const float *mass,
                                 const float *force_x,
                                 const float *force_y,
                                 const float *force_z,
                                 const float *vel_xsph_x,
                                 const float *vel_xsph_y,
                                 const float *vel_xsph_z,
                                 float       *vel_x_ping,
                                 float       *vel_x_pong,
                                 float       *vel_y_ping,
                                 float       *vel_y_pong,
                                 float       *vel_z_ping,
                                 float       *vel_z_pong,
                                 const int   *d_ping,
                                 const bool  *is_ghost,
                                 float       *pos_x,
                                 float       *pos_y,
                                 float       *pos_z,
                                 float       *out_force_x,
                                 float       *out_force_y,
                                 float       *out_force_z)
{
    int       i       = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_total = *d_n_total;
    if(i >= n_total)
        return;
    if(is_ghost[i])
    {
        out_force_x[i] = 0.0f;
        out_force_y[i] = 0.0f;
        out_force_z[i] = 0.0f;
        return;
    }

    const float *vel_x_read  = d_vel_read(vel_x_ping, vel_x_pong, d_ping);
    const float *vel_y_read  = d_vel_read(vel_y_ping, vel_y_pong, d_ping);
    const float *vel_z_read  = d_vel_read(vel_z_ping, vel_z_pong, d_ping);
    float       *vel_x_write = d_vel_write(vel_x_ping, vel_x_pong, d_ping);
    float       *vel_y_write = d_vel_write(vel_y_ping, vel_y_pong, d_ping);
    float       *vel_z_write = d_vel_write(vel_z_ping, vel_z_pong, d_ping);

    const float dt       = *d_dt;
    float       inv_mass = 1.0f / fmaxf(mass[i], MIN_RHO);
    float       ax       = force_x[i] * inv_mass;
    float       ay       = force_y[i] * inv_mass;
    float       az       = force_z[i] * inv_mass;
    float       v_old_x  = vel_x_read[i];
    float       v_old_y  = vel_y_read[i];
    float       v_old_z  = vel_z_read[i];
    float       v_new_x  = v_old_x + dt * ax;
    float       v_new_y  = v_old_y + dt * ay;
    float       v_new_z  = v_old_z + dt * az;

    vel_x_write[i] = v_new_x;
    vel_y_write[i] = v_new_y;
    vel_z_write[i] = v_new_z;

    pos_x[i] += (v_new_x + vel_xsph_x[i]) * dt;
    pos_y[i] += (v_new_y + vel_xsph_y[i]) * dt;
    pos_z[i] += (v_new_z + vel_xsph_z[i]) * dt;

    out_force_x[i] = 0.0f;
    out_force_y[i] = 0.0f;
    out_force_z[i] = 0.0f;
}

__global__ void kernel_boundary(const int *d_n_total,
                                float     *pos_x,
                                float     *pos_y,
                                float     *pos_z,
                                float     *vel_x_ping,
                                float     *vel_x_pong,
                                float     *vel_y_ping,
                                float     *vel_y_pong,
                                float     *vel_z_ping,
                                float     *vel_z_pong,
                                const bool *is_ghost,
                                const int *d_ping)
{
    int       i       = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_total = *d_n_total;
    if(i >= n_total)
        return;
    if(is_ghost[i])
        return;

    float *vel_x_active = d_vel_active(vel_x_ping, vel_x_pong, d_ping);
    float *vel_y_active = d_vel_active(vel_y_ping, vel_y_pong, d_ping);
    float *vel_z_active = d_vel_active(vel_z_ping, vel_z_pong, d_ping);

    float boundary_r  = SPHERE_R - PARTICLE_R;
    float boundary_r2 = boundary_r * boundary_r;
    float px          = pos_x[i];
    float py          = pos_y[i];
    float pz          = pos_z[i];
    float r2          = px * px + py * py + pz * pz;
    if(r2 < boundary_r2)
        return;

    const float wall_restitution = c_gpu_params.wall_restitution;
    const float friction         = c_gpu_params.friction;
    float       r                = sqrtf(r2);
    float       nx               = (r > 1e-12f) ? (px / r) : 0.0f;
    float       ny               = (r > 1e-12f) ? (py / r) : 0.0f;
    float       nz               = (r > 1e-12f) ? (pz / r) : 1.0f;
    px                           = boundary_r * nx;
    py                           = boundary_r * ny;
    pz                           = boundary_r * nz;

    float vx        = vel_x_active[i];
    float vy        = vel_y_active[i];
    float vz        = vel_z_active[i];
    float vn_scalar = vx * nx + vy * ny + vz * nz;
    if(vn_scalar > 0.0f)
    {
        float vnx                = vn_scalar * nx;
        float vny                = vn_scalar * ny;
        float vnz                = vn_scalar * nz;
        float vtx                = vx - vnx;
        float vty                = vy - vny;
        float vtz                = vz - vnz;
        float tangential_damping = 1.0f - friction;
        vx = (-wall_restitution) * vnx + tangential_damping * vtx;
        vy = (-wall_restitution) * vny + tangential_damping * vty;
        vz = (-wall_restitution) * vnz + tangential_damping * vtz;
    }

    pos_x[i]        = px;
    pos_y[i]        = py;
    pos_z[i]        = pz;
    vel_x_active[i] = vx;
    vel_y_active[i] = vy;
    vel_z_active[i] = vz;
}

__device__ void d_script_apply_key(unsigned char c, float *co)
{
    float       ax = co[0];
    float       ay = co[1];
    float       az = co[2];
    float       tx = co[3];
    float       ty = co[4];
    float       tz = co[5];
    const float as = c_shake_ui.accel_step;
    const float ts = c_shake_ui.torque_step;
    const float al = c_shake_ui.accel_limit;
    const float tl = c_shake_ui.torque_limit;

    auto bump_axis = [&](float &axis, float delta)
    {
        axis += delta;
        axis = fmaxf(-al, fminf(axis, al));
    };
    auto bump_torque = [&](float &axis, float delta)
    {
        axis += delta;
        axis = fmaxf(-tl, fminf(axis, tl));
    };

    if(c == 'w' || c == 'W')
        bump_axis(ay, as);
    else if(c == 's' || c == 'S')
        bump_axis(ay, -as);
    else if(c == 'a' || c == 'A')
        bump_axis(ax, -as);
    else if(c == 'd' || c == 'D')
        bump_axis(ax, as);
    else if(c == 'q' || c == 'Q')
        bump_axis(az, as);
    else if(c == 'e' || c == 'E')
        bump_axis(az, -as);
    else if(c == 'i' || c == 'I')
        bump_torque(tx, ts);
    else if(c == 'k' || c == 'K')
        bump_torque(tx, -ts);
    else if(c == 'j' || c == 'J')
        bump_torque(ty, ts);
    else if(c == 'l' || c == 'L')
        bump_torque(ty, -ts);
    else if(c == 'u' || c == 'U')
        bump_torque(tz, ts);
    else if(c == 'o' || c == 'O')
        bump_torque(tz, -ts);
    else if(c == ' ')
    {
        ax = ay = az = 0.0f;
        tx = ty = tz = 0.0f;
    }

    co[0] = ax;
    co[1] = ay;
    co[2] = az;
    co[3] = tx;
    co[4] = ty;
    co[5] = tz;
}

} // namespace

bool launch_density_pressure_forces(SimState &s, SimGpuBuffers &buffers)
{
    constexpr int block_size = 256;
    if(s.n_fluid <= 0)
        return true;

    const int grid = div_up(N_MAX, block_size);

    kernel_hash_particles<<<grid, block_size>>>(buffers.d_n_total,
                                                buffers.pos_x,
                                                buffers.pos_y,
                                                buffers.pos_z,
                                                buffers.radix_key,
                                                buffers.sorted_id);
    if(!launch_ok())
        return false;

    cudaError_t sort_err = cub::DeviceRadixSort::SortPairs(
        buffers.cub_sort_temp,
        buffers.cub_sort_temp_bytes,
        reinterpret_cast<std::uint32_t *>(buffers.radix_key),
        reinterpret_cast<std::uint32_t *>(buffers.radix_key_alt),
        buffers.sorted_id,
        buffers.sorted_id_alt,
        N_MAX,
        0,
        10);
    if(sort_err != cudaSuccess)
        return false;

    std::swap(buffers.radix_key, buffers.radix_key_alt);
    std::swap(buffers.sorted_id, buffers.sorted_id_alt);

    kernel_reorder_particle_arrays<<<grid, block_size>>>(
        buffers.d_n_total,
        buffers.sorted_id,
        buffers.pos_x,
        buffers.pos_y,
        buffers.pos_z,
        buffers.vel_x[PING],
        buffers.vel_y[PING],
        buffers.vel_z[PING],
        buffers.vel_x[PONG],
        buffers.vel_y[PONG],
        buffers.vel_z[PONG],
        buffers.vel_xsph_x,
        buffers.vel_xsph_y,
        buffers.vel_xsph_z,
        buffers.mass,
        buffers.density,
        buffers.pressure,
        buffers.is_ghost,
        buffers.particle_id,
        buffers.pos_x_sorted,
        buffers.pos_y_sorted,
        buffers.pos_z_sorted,
        buffers.vel_x_sorted[PING],
        buffers.vel_y_sorted[PING],
        buffers.vel_z_sorted[PING],
        buffers.vel_x_sorted[PONG],
        buffers.vel_y_sorted[PONG],
        buffers.vel_z_sorted[PONG],
        buffers.vel_xsph_x_sorted,
        buffers.vel_xsph_y_sorted,
        buffers.vel_xsph_z_sorted,
        buffers.mass_sorted,
        buffers.density_sorted,
        buffers.pressure_sorted,
        buffers.is_ghost_sorted,
        buffers.particle_id_sorted);
    if(!launch_ok())
        return false;

    std::swap(buffers.pos_x, buffers.pos_x_sorted);
    std::swap(buffers.pos_y, buffers.pos_y_sorted);
    std::swap(buffers.pos_z, buffers.pos_z_sorted);
    std::swap(buffers.vel_x[PING], buffers.vel_x_sorted[PING]);
    std::swap(buffers.vel_y[PING], buffers.vel_y_sorted[PING]);
    std::swap(buffers.vel_z[PING], buffers.vel_z_sorted[PING]);
    std::swap(buffers.vel_x[PONG], buffers.vel_x_sorted[PONG]);
    std::swap(buffers.vel_y[PONG], buffers.vel_y_sorted[PONG]);
    std::swap(buffers.vel_z[PONG], buffers.vel_z_sorted[PONG]);
    std::swap(buffers.vel_xsph_x, buffers.vel_xsph_x_sorted);
    std::swap(buffers.vel_xsph_y, buffers.vel_xsph_y_sorted);
    std::swap(buffers.vel_xsph_z, buffers.vel_xsph_z_sorted);
    std::swap(buffers.mass, buffers.mass_sorted);
    std::swap(buffers.density, buffers.density_sorted);
    std::swap(buffers.pressure, buffers.pressure_sorted);
    std::swap(buffers.is_ghost, buffers.is_ghost_sorted);
    std::swap(buffers.particle_id, buffers.particle_id_sorted);
    kernel_reset_sorted_id<<<grid, block_size>>>(buffers.d_n_total,
                                                 buffers.sorted_id);
    if(!launch_ok())
        return false;

    if(cudaMemset(buffers.cell_start, 0xff, sizeof(int) * TABLE_SIZE) !=
           cudaSuccess ||
       cudaMemset(buffers.cell_count, 0, sizeof(int) * TABLE_SIZE) !=
       cudaSuccess)
        return false;

    kernel_fill_cell_ranges<<<grid, block_size>>>(
        buffers.d_n_total, buffers.radix_key, buffers.cell_start, buffers.cell_count);
    if(!launch_ok())
        return false;
    kernel_build_neighbor_lists<<<grid, block_size>>>(
        buffers.d_n_total,
        buffers.pos_x,
        buffers.pos_y,
        buffers.pos_z,
        buffers.sorted_id,
        buffers.radix_key,
        buffers.cell_start,
        buffers.cell_count,
        buffers.is_ghost,
        buffers.neighbor_list,
        buffers.neighbor_count);
    kernel_compute_density_pressure<<<grid, block_size>>>(
        buffers.d_n_total,
        buffers.neighbor_count,
        buffers.neighbor_list,
        buffers.pos_x,
        buffers.pos_y,
        buffers.pos_z,
        buffers.mass,
        buffers.is_ghost,
        buffers.density,
        buffers.pressure);
    kernel_compute_forces<<<grid, block_size>>>(buffers.d_n_total,
                                                buffers.neighbor_count,
                                                buffers.neighbor_list,
                                                buffers.pos_x,
                                                buffers.pos_y,
                                                buffers.pos_z,
                                                buffers.vel_x[PING],
                                                buffers.vel_x[PONG],
                                                buffers.vel_y[PING],
                                                buffers.vel_y[PONG],
                                                buffers.vel_z[PING],
                                                buffers.vel_z[PONG],
                                                buffers.d_ping,
                                                buffers.mass,
                                                buffers.density,
                                                buffers.pressure,
                                                buffers.is_ghost,
                                                buffers.force_x,
                                                buffers.force_y,
                                                buffers.force_z,
                                                buffers.vel_xsph_x,
                                                buffers.vel_xsph_y,
                                                buffers.vel_xsph_z);
    return launch_ok();
}

bool launch_density_pressure_forces_reuse_neighbors(SimState &s,
                                                    SimGpuBuffers &buffers)
{
    constexpr int block_size = 256;
    if(s.n_fluid <= 0)
        return true;
    const int grid = div_up(N_MAX, block_size);
    kernel_compute_density_pressure<<<grid, block_size>>>(
        buffers.d_n_total,
        buffers.neighbor_count,
        buffers.neighbor_list,
        buffers.pos_x,
        buffers.pos_y,
        buffers.pos_z,
        buffers.mass,
        buffers.is_ghost,
        buffers.density,
        buffers.pressure);
    if(!launch_ok())
        return false;
    kernel_compute_forces<<<grid, block_size>>>(buffers.d_n_total,
                                                buffers.neighbor_count,
                                                buffers.neighbor_list,
                                                buffers.pos_x,
                                                buffers.pos_y,
                                                buffers.pos_z,
                                                buffers.vel_x[PING],
                                                buffers.vel_x[PONG],
                                                buffers.vel_y[PING],
                                                buffers.vel_y[PONG],
                                                buffers.vel_z[PING],
                                                buffers.vel_z[PONG],
                                                buffers.d_ping,
                                                buffers.mass,
                                                buffers.density,
                                                buffers.pressure,
                                                buffers.is_ghost,
                                                buffers.force_x,
                                                buffers.force_y,
                                                buffers.force_z,
                                                buffers.vel_xsph_x,
                                                buffers.vel_xsph_y,
                                                buffers.vel_xsph_z);
    return launch_ok();
}

bool launch_density_pressure_forces_fused(SimState &s, SimGpuBuffers &buffers)
{
    // Current fused path reuses the optimized standard path while preserving
    // the runtime switch contract.
    return launch_density_pressure_forces(s, buffers);
}

bool launch_init_gpu_particle_ids(int n_fluid, SimGpuBuffers &buffers)
{
    if(n_fluid <= 0)
        return true;
    constexpr int block_size = 256;
    const int     grid       = div_up(N_MAX, block_size);
    kernel_init_particle_ids<<<grid, block_size>>>(n_fluid, buffers.particle_id);
    return launch_ok();
}

bool launch_inject_ghost_particles(SimState &s, SimGpuBuffers &buffers)
{
    constexpr int block_size = 256;
    if(s.n_fluid <= 0)
        return cudaMemset(buffers.d_n_total, 0, sizeof(int)) == cudaSuccess;

    const int grid_all = div_up(N_MAX, block_size);
    kernel_fill_fluid_compact_stencil<<<grid_all, block_size>>>(
        buffers.d_n_total, s.n_fluid, buffers.is_ghost, buffers.ghost_flags);
    if(!launch_ok())
        return false;

    thrust::exclusive_scan(thrust::device,
                           buffers.ghost_flags,
                           buffers.ghost_flags + N_MAX,
                           buffers.ghost_offsets);

    kernel_compact_real_fluid_to_prefix<<<grid_all, block_size>>>(
        buffers.d_n_total,
        s.n_fluid,
        buffers.ghost_offsets,
        buffers.pos_x,
        buffers.pos_y,
        buffers.pos_z,
        buffers.vel_x[PING],
        buffers.vel_y[PING],
        buffers.vel_z[PING],
        buffers.vel_x[PONG],
        buffers.vel_y[PONG],
        buffers.vel_z[PONG],
        buffers.vel_xsph_x,
        buffers.vel_xsph_y,
        buffers.vel_xsph_z,
        buffers.mass,
        buffers.density,
        buffers.pressure,
        buffers.is_ghost,
        buffers.particle_id,
        buffers.pos_x_sorted,
        buffers.pos_y_sorted,
        buffers.pos_z_sorted,
        buffers.vel_x_sorted[PING],
        buffers.vel_y_sorted[PING],
        buffers.vel_z_sorted[PING],
        buffers.vel_x_sorted[PONG],
        buffers.vel_y_sorted[PONG],
        buffers.vel_z_sorted[PONG],
        buffers.vel_xsph_x_sorted,
        buffers.vel_xsph_y_sorted,
        buffers.vel_xsph_z_sorted,
        buffers.mass_sorted,
        buffers.density_sorted,
        buffers.pressure_sorted,
        buffers.is_ghost_sorted,
        buffers.particle_id_sorted);
    if(!launch_ok())
        return false;

    std::swap(buffers.pos_x, buffers.pos_x_sorted);
    std::swap(buffers.pos_y, buffers.pos_y_sorted);
    std::swap(buffers.pos_z, buffers.pos_z_sorted);
    std::swap(buffers.vel_x[PING], buffers.vel_x_sorted[PING]);
    std::swap(buffers.vel_y[PING], buffers.vel_y_sorted[PING]);
    std::swap(buffers.vel_z[PING], buffers.vel_z_sorted[PING]);
    std::swap(buffers.vel_x[PONG], buffers.vel_x_sorted[PONG]);
    std::swap(buffers.vel_y[PONG], buffers.vel_y_sorted[PONG]);
    std::swap(buffers.vel_z[PONG], buffers.vel_z_sorted[PONG]);
    std::swap(buffers.vel_xsph_x, buffers.vel_xsph_x_sorted);
    std::swap(buffers.vel_xsph_y, buffers.vel_xsph_y_sorted);
    std::swap(buffers.vel_xsph_z, buffers.vel_xsph_z_sorted);
    std::swap(buffers.mass, buffers.mass_sorted);
    std::swap(buffers.density, buffers.density_sorted);
    std::swap(buffers.pressure, buffers.pressure_sorted);
    std::swap(buffers.is_ghost, buffers.is_ghost_sorted);
    std::swap(buffers.particle_id, buffers.particle_id_sorted);

    const int grid = div_up(std::max(s.n_fluid, MAX_DIE_GHOSTS), block_size);
    kernel_mark_ghost_particles<<<grid, block_size>>>(s.n_fluid,
                                                      buffers.pos_x,
                                                      buffers.pos_y,
                                                      buffers.pos_z,
                                                      buffers.ghost_flags);
    if(!launch_ok())
        return false;

    thrust::exclusive_scan(thrust::device,
                           buffers.ghost_flags,
                           buffers.ghost_flags + s.n_fluid,
                           buffers.ghost_offsets);

    kernel_scatter_ghost_particles<<<grid, block_size>>>(
        s.n_fluid,
        buffers.pos_x,
        buffers.pos_y,
        buffers.pos_z,
        buffers.vel_x[PING],
        buffers.vel_y[PING],
        buffers.vel_z[PING],
        buffers.vel_x[PONG],
        buffers.vel_y[PONG],
        buffers.vel_z[PONG],
        buffers.vel_xsph_x,
        buffers.vel_xsph_y,
        buffers.vel_xsph_z,
        buffers.mass,
        buffers.density,
        buffers.pressure,
        buffers.is_ghost,
        buffers.particle_id,
        buffers.ghost_flags,
        buffers.ghost_offsets,
        buffers.d_die,
        buffers.d_n_total);
    return launch_ok();
}

bool launch_gather_canonical_frame_layout(int n_fluid, int n_total,
                                          SimGpuBuffers &buffers)
{
    if(n_fluid <= 0 || n_total <= 0)
        return true;
    constexpr int block_size = 256;
    const int     grid_nt    = div_up(n_total, block_size);

    kernel_scatter_fluid_fields_by_id<<<grid_nt, block_size>>>(
        n_total,
        n_fluid,
        buffers.pos_x,
        buffers.pos_y,
        buffers.pos_z,
        buffers.density,
        buffers.is_ghost,
        buffers.particle_id,
        buffers.pos_x_sorted,
        buffers.pos_y_sorted,
        buffers.pos_z_sorted,
        buffers.density_sorted);

    kernel_fill_ghost_gather_stencil<<<grid_nt, block_size>>>(
        n_total, buffers.is_ghost, buffers.ghost_flags);
    if(!launch_ok())
        return false;

    thrust::exclusive_scan(thrust::device,
                           buffers.ghost_flags,
                           buffers.ghost_flags + n_total,
                           buffers.ghost_offsets);

    kernel_scatter_ghost_positions_to_tail<<<grid_nt, block_size>>>(
        n_total,
        n_fluid,
        buffers.ghost_offsets,
        buffers.pos_x,
        buffers.pos_y,
        buffers.pos_z,
        buffers.is_ghost,
        buffers.pos_x_sorted,
        buffers.pos_y_sorted,
        buffers.pos_z_sorted);
    return launch_ok();
}

bool launch_die_coupling(SimState &s, SimGpuBuffers &buffers)
{
    (void)s;
    constexpr int block_size = 256;
    int           grid       = div_up(N_MAX, block_size);
    kernel_die_coupling<<<grid, block_size>>>(buffers.d_n_total,
                                              buffers.d_die,
                                              buffers.pos_x,
                                              buffers.pos_y,
                                              buffers.pos_z,
                                              buffers.vel_x[PING],
                                              buffers.vel_x[PONG],
                                              buffers.vel_y[PING],
                                              buffers.vel_y[PONG],
                                              buffers.vel_z[PING],
                                              buffers.vel_z[PONG],
                                              buffers.d_ping,
                                              buffers.is_ghost,
                                              buffers.density,
                                              buffers.force_x,
                                              buffers.force_y,
                                              buffers.force_z,
                                              buffers.die_force,
                                              buffers.die_torque);
    return launch_ok();
}

bool launch_dt_reduction(SimState &s, SimGpuBuffers &buffers)
{
    (void)s;
    int grid = DT_REDUCTION_SCRATCH;
    kernel_dt_block_max<<<grid, DT_REDUCTION_BLOCK_SIZE>>>(
        buffers.d_n_total,
        buffers.vel_x[PING],
        buffers.vel_x[PONG],
        buffers.vel_y[PING],
        buffers.vel_y[PONG],
        buffers.vel_z[PING],
        buffers.vel_z[PONG],
        buffers.d_ping,
        buffers.is_ghost,
        buffers.force_x,
        buffers.force_y,
        buffers.force_z,
        buffers.mass,
        buffers.dt_block_max_speed,
        buffers.dt_block_max_accel);
    return launch_ok();
}

bool launch_reduce_dt_and_finalize(SimState &s, SimGpuBuffers &buffers)
{
    (void)s;
    const int block_count = DT_REDUCTION_SCRATCH;
    kernel_reduce_dt_and_finalize<<<1, 1>>>(block_count,
                                            buffers.dt_block_max_speed,
                                            buffers.dt_block_max_accel,
                                            buffers.d_max_speed,
                                            buffers.d_max_accel,
                                            buffers.d_dt,
                                            buffers.d_dt_diagnostic,
                                            buffers.d_dt_limiter);
    return launch_ok();
}

bool launch_integrate(SimState &s, SimGpuBuffers &buffers)
{
    (void)s;
    constexpr int block_size = 256;
    int           grid       = div_up(N_MAX, block_size);
    kernel_integrate<<<grid, block_size>>>(buffers.d_n_total,
                                           buffers.d_dt,
                                           buffers.mass,
                                           buffers.force_x,
                                           buffers.force_y,
                                           buffers.force_z,
                                           buffers.vel_xsph_x,
                                           buffers.vel_xsph_y,
                                           buffers.vel_xsph_z,
                                           buffers.vel_x[PING],
                                           buffers.vel_x[PONG],
                                           buffers.vel_y[PING],
                                           buffers.vel_y[PONG],
                                           buffers.vel_z[PING],
                                           buffers.vel_z[PONG],
                                           buffers.d_ping,
                                           buffers.is_ghost,
                                           buffers.pos_x,
                                           buffers.pos_y,
                                           buffers.pos_z,
                                           buffers.force_x,
                                           buffers.force_y,
                                           buffers.force_z);
    return launch_ok();
}

bool launch_boundary(SimState &s, SimGpuBuffers &buffers)
{
    (void)s;
    constexpr int block_size = 256;
    int           grid       = div_up(N_MAX, block_size);
    kernel_boundary<<<grid, block_size>>>(buffers.d_n_total,
                                          buffers.pos_x,
                                          buffers.pos_y,
                                          buffers.pos_z,
                                          buffers.vel_x[PING],
                                          buffers.vel_x[PONG],
                                          buffers.vel_y[PING],
                                          buffers.vel_y[PONG],
                                          buffers.vel_z[PING],
                                          buffers.vel_z[PONG],
                                          buffers.is_ghost,
                                          buffers.d_ping);
    return launch_ok();
}

bool launch_toggle_ping(SimGpuBuffers &buffers)
{
    kernel_toggle_ping<<<1, 1>>>(buffers.d_ping);
    return launch_ok();
}

bool launch_integrate_rigid_die(SimGpuBuffers             &buffers,
                                const SimGpuPendingInput &pending_input)
{
    kernel_integrate_rigid_die<<<1, 1>>>(buffers.d_die,
                                         buffers.d_dt,
                                         buffers.d_control,
                                         buffers.die_force,
                                         buffers.die_torque,
                                         buffers.die_clamp_debug,
                                         buffers.d_sim_time,
                                         buffers.d_step,
                                         buffers.d_prev_dt,
                                         pending_input,
                                         buffers.d_script_times,
                                         buffers.d_script_keys,
                                         buffers.d_script_count,
                                         buffers.d_script_cursor);
    return launch_ok();
}

bool sim_cuda_upload_constant_params(const Params &params)
{
    return cudaMemcpyToSymbol(c_gpu_params, &params, sizeof(Params)) ==
           cudaSuccess;
}

bool sim_cuda_upload_constant_runtime_caps(const SimGpuRuntimeCaps &caps)
{
    return cudaMemcpyToSymbol(
               c_runtime_caps, &caps, sizeof(SimGpuRuntimeCaps)) == cudaSuccess;
}

bool sim_cuda_upload_constant_shake_ui(const ShakeUiKernels &k)
{
    return cudaMemcpyToSymbol(c_shake_ui, &k, sizeof(ShakeUiKernels)) ==
           cudaSuccess;
}
