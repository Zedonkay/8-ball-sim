#include "constants.h"
#include "sim_cuda_internal.h"

#include <cstdio>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

int div_up(int n, int d)
{
    return (n + d - 1) / d;
}

bool alloc_buffers(SimGpuBuffers &b)
{
    auto alloc = [](auto **ptr, std::size_t nbytes, const char *name) -> bool
    {
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(ptr), nbytes);
        if(err != cudaSuccess)
        {
            std::fprintf(stderr,
                         "cudaMalloc failed for %s (%zu bytes): %s\n",
                         name,
                         static_cast<std::size_t>(nbytes),
                         cudaGetErrorString(err));
            return false;
        }
        return true;
    };

    if(!alloc(&b.pos_x, sizeof(float) * N_MAX, "pos_x") ||
       !alloc(&b.pos_y, sizeof(float) * N_MAX, "pos_y") ||
       !alloc(&b.pos_z, sizeof(float) * N_MAX, "pos_z") ||
       !alloc(&b.pos_x_sorted, sizeof(float) * N_MAX, "pos_x_sorted") ||
       !alloc(&b.pos_y_sorted, sizeof(float) * N_MAX, "pos_y_sorted") ||
       !alloc(&b.pos_z_sorted, sizeof(float) * N_MAX, "pos_z_sorted") ||
       !alloc(&b.vel_x[PING], sizeof(float) * N_MAX, "vel_x[PING]") ||
       !alloc(&b.vel_y[PING], sizeof(float) * N_MAX, "vel_y[PING]") ||
       !alloc(&b.vel_z[PING], sizeof(float) * N_MAX, "vel_z[PING]") ||
       !alloc(&b.vel_x[PONG], sizeof(float) * N_MAX, "vel_x[PONG]") ||
       !alloc(&b.vel_y[PONG], sizeof(float) * N_MAX, "vel_y[PONG]") ||
       !alloc(&b.vel_z[PONG], sizeof(float) * N_MAX, "vel_z[PONG]") ||
       !alloc(&b.vel_x_sorted[PING], sizeof(float) * N_MAX, "vel_x_sorted[PING]") ||
       !alloc(&b.vel_y_sorted[PING], sizeof(float) * N_MAX, "vel_y_sorted[PING]") ||
       !alloc(&b.vel_z_sorted[PING], sizeof(float) * N_MAX, "vel_z_sorted[PING]") ||
       !alloc(&b.vel_x_sorted[PONG], sizeof(float) * N_MAX, "vel_x_sorted[PONG]") ||
       !alloc(&b.vel_y_sorted[PONG], sizeof(float) * N_MAX, "vel_y_sorted[PONG]") ||
       !alloc(&b.vel_z_sorted[PONG], sizeof(float) * N_MAX, "vel_z_sorted[PONG]") ||
       !alloc(&b.vel_xsph_x, sizeof(float) * N_MAX, "vel_xsph_x") ||
       !alloc(&b.vel_xsph_y, sizeof(float) * N_MAX, "vel_xsph_y") ||
       !alloc(&b.vel_xsph_z, sizeof(float) * N_MAX, "vel_xsph_z") ||
       !alloc(&b.vel_xsph_x_sorted, sizeof(float) * N_MAX, "vel_xsph_x_sorted") ||
       !alloc(&b.vel_xsph_y_sorted, sizeof(float) * N_MAX, "vel_xsph_y_sorted") ||
       !alloc(&b.vel_xsph_z_sorted, sizeof(float) * N_MAX, "vel_xsph_z_sorted") ||
       !alloc(&b.force_x, sizeof(float) * N_MAX, "force_x") ||
       !alloc(&b.force_y, sizeof(float) * N_MAX, "force_y") ||
       !alloc(&b.force_z, sizeof(float) * N_MAX, "force_z") ||
       !alloc(&b.mass, sizeof(float) * N_MAX, "mass") ||
       !alloc(&b.mass_sorted, sizeof(float) * N_MAX, "mass_sorted") ||
       !alloc(&b.density, sizeof(float) * N_MAX, "density") ||
       !alloc(&b.density_sorted, sizeof(float) * N_MAX, "density_sorted") ||
       !alloc(&b.pressure, sizeof(float) * N_MAX, "pressure") ||
       !alloc(&b.pressure_sorted, sizeof(float) * N_MAX, "pressure_sorted") ||
       !alloc(&b.is_ghost, sizeof(bool) * N_MAX, "is_ghost") ||
       !alloc(&b.is_ghost_sorted, sizeof(bool) * N_MAX, "is_ghost_sorted") ||
       !alloc(&b.particle_id, sizeof(int) * N_MAX, "particle_id") ||
       !alloc(&b.particle_id_sorted, sizeof(int) * N_MAX, "particle_id_sorted") ||
       !alloc(&b.radix_key, sizeof(std::uint32_t) * N_MAX, "radix_key") ||
       !alloc(
           &b.radix_key_alt, sizeof(std::uint32_t) * N_MAX, "radix_key_alt") ||
       !alloc(&b.sorted_id, sizeof(int) * N_MAX, "sorted_id") ||
       !alloc(&b.sorted_id_alt, sizeof(int) * N_MAX, "sorted_id_alt") ||
       !alloc(&b.cell_start, sizeof(int) * TABLE_SIZE, "cell_start") ||
       !alloc(&b.cell_count, sizeof(int) * TABLE_SIZE, "cell_count") ||
       !alloc(&b.neighbor_list,
              sizeof(int) * N_MAX * MAX_NEIGHBORS,
              "neighbor_list") ||
       !alloc(&b.neighbor_count, sizeof(int) * N_MAX, "neighbor_count") ||
       !alloc(&b.ghost_flags, sizeof(int) * N_MAX, "ghost_flags") ||
       !alloc(&b.ghost_offsets, sizeof(int) * N_MAX, "ghost_offsets") ||
       !alloc(&b.dt_block_max_speed,
              sizeof(float) * DT_REDUCTION_SCRATCH,
              "dt_block_max_speed") ||
       !alloc(&b.dt_block_max_accel,
              sizeof(float) * DT_REDUCTION_SCRATCH,
              "dt_block_max_accel") ||
       !alloc(&b.die_force, sizeof(float) * 3, "die_force") ||
       !alloc(&b.die_torque, sizeof(float) * 3, "die_torque") ||
       !alloc(&b.die_clamp_debug, sizeof(float) * 12, "die_clamp_debug") ||
       !alloc(&b.d_n_total, sizeof(int), "d_n_total") ||
       !alloc(&b.d_ping, sizeof(int), "d_ping") ||
       !alloc(&b.d_dt, sizeof(float), "d_dt") ||
       !alloc(&b.d_max_speed, sizeof(float), "d_max_speed") ||
       !alloc(&b.d_max_accel, sizeof(float), "d_max_accel") ||
       !alloc(&b.d_dt_diagnostic, sizeof(float) * 7, "d_dt_diagnostic") ||
       !alloc(&b.d_dt_limiter, sizeof(int), "d_dt_limiter") ||
       !alloc(&b.d_die, sizeof(GpuRigidDie), "d_die") ||
       !alloc(&b.d_control, sizeof(float) * 6, "d_control") ||
       !alloc(&b.d_sim_time, sizeof(float), "d_sim_time") ||
       !alloc(&b.d_step, sizeof(int), "d_step") ||
       !alloc(&b.d_prev_dt, sizeof(float), "d_prev_dt") ||
       !alloc(&b.d_script_count, sizeof(int), "d_script_count") ||
       !alloc(&b.d_script_cursor, sizeof(int), "d_script_cursor"))
    {
        return false;
    }

    std::size_t cub_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        cub_bytes,
        reinterpret_cast<std::uint32_t *>(b.radix_key),
        reinterpret_cast<std::uint32_t *>(b.radix_key_alt),
        b.sorted_id,
        b.sorted_id_alt,
        N_MAX);
    if(!alloc(&b.cub_sort_temp, cub_bytes, "cub_sort_temp"))
    {
        return false;
    }
    b.cub_sort_temp_bytes = cub_bytes;

    if(cudaMallocHost(reinterpret_cast<void **>(&b.h_pos_x), sizeof(float) * N_MAX) !=
           cudaSuccess ||
       cudaMallocHost(reinterpret_cast<void **>(&b.h_pos_y), sizeof(float) * N_MAX) !=
           cudaSuccess ||
       cudaMallocHost(reinterpret_cast<void **>(&b.h_pos_z), sizeof(float) * N_MAX) !=
           cudaSuccess ||
       cudaMallocHost(reinterpret_cast<void **>(&b.h_density),
                      sizeof(float) * N_MAX) != cudaSuccess)
    {
        return false;
    }

    if(cudaStreamCreate(&b.stream_compute) != cudaSuccess ||
       cudaStreamCreate(&b.stream_copy) != cudaSuccess)
    {
        return false;
    }
    if(cudaEventCreateWithFlags(&b.evt_copy_ready, cudaEventDisableTiming) !=
       cudaSuccess)
    {
        return false;
    }

    if(cudaMemset(b.d_sim_time, 0, sizeof(float)) != cudaSuccess ||
       cudaMemset(b.d_step, 0, sizeof(int)) != cudaSuccess ||
       cudaMemset(b.d_control, 0, sizeof(float) * 6) != cudaSuccess)
    {
        return false;
    }
    const float prev0 = 0.0f;
    if(cudaMemcpy(b.d_prev_dt, &prev0, sizeof(float), cudaMemcpyHostToDevice) !=
       cudaSuccess)
    {
        return false;
    }
    const int script_n0 = 0;
    if(cudaMemcpy(
           b.d_script_count, &script_n0, sizeof(int), cudaMemcpyHostToDevice) !=
       cudaSuccess)
    {
        return false;
    }
    const int zc = 0;
    if(cudaMemcpy(
           b.d_script_cursor, &zc, sizeof(int), cudaMemcpyHostToDevice) !=
       cudaSuccess)
    {
        return false;
    }

    b.d_script_times = nullptr;
    b.d_script_keys  = nullptr;

    b.initialized = true;
    return true;
}

void free_buffers(SimGpuBuffers &b)
{
    cudaFree(b.pos_x);
    cudaFree(b.pos_y);
    cudaFree(b.pos_z);
    cudaFree(b.pos_x_sorted);
    cudaFree(b.pos_y_sorted);
    cudaFree(b.pos_z_sorted);
    cudaFree(b.vel_x[PING]);
    cudaFree(b.vel_y[PING]);
    cudaFree(b.vel_z[PING]);
    cudaFree(b.vel_x[PONG]);
    cudaFree(b.vel_y[PONG]);
    cudaFree(b.vel_z[PONG]);
    cudaFree(b.vel_x_sorted[PING]);
    cudaFree(b.vel_y_sorted[PING]);
    cudaFree(b.vel_z_sorted[PING]);
    cudaFree(b.vel_x_sorted[PONG]);
    cudaFree(b.vel_y_sorted[PONG]);
    cudaFree(b.vel_z_sorted[PONG]);
    cudaFree(b.vel_xsph_x);
    cudaFree(b.vel_xsph_y);
    cudaFree(b.vel_xsph_z);
    cudaFree(b.vel_xsph_x_sorted);
    cudaFree(b.vel_xsph_y_sorted);
    cudaFree(b.vel_xsph_z_sorted);
    cudaFree(b.force_x);
    cudaFree(b.force_y);
    cudaFree(b.force_z);
    cudaFree(b.mass);
    cudaFree(b.mass_sorted);
    cudaFree(b.density);
    cudaFree(b.density_sorted);
    cudaFree(b.pressure);
    cudaFree(b.pressure_sorted);
    cudaFree(b.is_ghost);
    cudaFree(b.is_ghost_sorted);
    cudaFree(b.particle_id);
    cudaFree(b.particle_id_sorted);
    cudaFree(b.radix_key);
    cudaFree(b.radix_key_alt);
    cudaFree(b.sorted_id);
    cudaFree(b.sorted_id_alt);
    cudaFree(b.cell_start);
    cudaFree(b.cell_count);
    cudaFree(b.neighbor_list);
    cudaFree(b.neighbor_count);
    cudaFree(b.ghost_flags);
    cudaFree(b.ghost_offsets);
    cudaFree(b.dt_block_max_speed);
    cudaFree(b.dt_block_max_accel);
    cudaFree(b.die_force);
    cudaFree(b.die_torque);
    cudaFree(b.die_clamp_debug);
    cudaFree(b.d_n_total);
    cudaFree(b.d_ping);
    cudaFree(b.d_dt);
    cudaFree(b.d_max_speed);
    cudaFree(b.d_max_accel);
    cudaFree(b.d_dt_diagnostic);
    cudaFree(b.d_dt_limiter);
    cudaFree(b.d_die);
    cudaFree(b.d_control);
    cudaFree(b.d_sim_time);
    cudaFree(b.d_step);
    cudaFree(b.d_prev_dt);
    cudaFree(b.d_script_times);
    cudaFree(b.d_script_keys);
    cudaFree(b.d_script_count);
    cudaFree(b.d_script_cursor);
    cudaFree(b.cub_sort_temp);
    cudaFreeHost(b.h_pos_x);
    cudaFreeHost(b.h_pos_y);
    cudaFreeHost(b.h_pos_z);
    cudaFreeHost(b.h_density);
    if(b.evt_copy_ready)
        cudaEventDestroy(b.evt_copy_ready);
    if(b.stream_compute)
        cudaStreamDestroy(b.stream_compute);
    if(b.stream_copy)
        cudaStreamDestroy(b.stream_copy);
    b = SimGpuBuffers{};
}

bool copy_to_device(void *dst, const void *src, std::size_t bytes)
{
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool copy_to_host(void *dst, const void *src, std::size_t bytes)
{
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool launch_ok()
{
    cudaError_t launch_err = cudaGetLastError();
    if(launch_err != cudaSuccess)
    {
        std::fprintf(stderr,
                     "cuda kernel launch failed: %s\n",
                     cudaGetErrorString(launch_err));
        return false;
    }
    return true;
}

bool sync_ok()
{
    if(!launch_ok())
        return false;
    if(!SIM_GPU_STRICT_SYNC)
        return true;
    cudaError_t sync_err = cudaDeviceSynchronize();
    if(sync_err != cudaSuccess)
    {
        std::fprintf(stderr,
                     "cuda synchronize failed: %s\n",
                     cudaGetErrorString(sync_err));
        return false;
    }
    return true;
}
