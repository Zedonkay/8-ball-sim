#pragma once

#include "constants.h"

#define PARTICLE_TYPE_FLUID      0
#define PARTICLE_TYPE_GHOST_WALL 1
#define PARTICLE_TYPE_DIE_WALL   2

struct SimState
{
    // ── Position
    // ─────────────────────────────────────────────────────────────
    float pos_x[N_MAX]{};
    float pos_y[N_MAX]{};
    float pos_z[N_MAX]{};

    // ── Velocity (double-buffered: read ping, write pong, swap each step)
    // ──── first index is buffer selector: [PING] or [PONG].
    float vel_x[2][N_MAX]{};
    float vel_y[2][N_MAX]{};
    float vel_z[2][N_MAX]{};

    // ── XSPH position-advection correction (written by force kernel, added to
    // the newly integrated velocity) ─
    float vel_xsph_x[N_MAX]{};
    float vel_xsph_y[N_MAX]{};
    float vel_xsph_z[N_MAX]{};

    // ── Accumulated force (zeroed at end of integration each step)
    // ────────────
    float force_x[N_MAX]{};
    float force_y[N_MAX]{};
    float force_z[N_MAX]{};

    // ── Per-particle scalars
    // ──────────────────────────────────────────────────
    float mass[N_MAX]{};
    float density[N_MAX]{};
    float pressure[N_MAX]{};

    // ── Particle classification
    // ─────────────────────────────────────────────── 0 = fluid,  1 =
    // sphere-wall ghost,  2 = die ghost (reserved)
    int  particle_type[N_MAX]{};
    bool is_ghost[N_MAX]{};

    // CPU spatial-hash path compatibility (used by spatial_hash.cpp in
    // non-GPU/legacy builds).
    int neighbor_list[N_MAX * MAX_NEIGHBORS]{};
    int neighbor_count[N_MAX]{};
    int hash_key[N_MAX]{};
    int cell_ix[N_MAX]{};
    int cell_iy[N_MAX]{};
    int cell_iz[N_MAX]{};
    int sorted_id[N_MAX]{};
    int cell_start[TABLE_SIZE]{};
    int cell_count[TABLE_SIZE]{};

    // ── Runtime counters
    // ──────────────────────────────────────────────────────
    int n_fluid{}; // fluid-only particles (constant after initialisation)
    int n_total{}; // fluid + ghost;

    // ── Ping-pong buffer selector
    // ─────────────────────────────────────────────
    int ping{PING};
    int pong{PONG};

    void swap_buffers()
    {
        int tmp = ping;
        ping    = pong;
        pong    = tmp;
    }

    // ── parameters ──────────────────────────────────────────
    Params params{};

    // Opaque GPU device buffers (SimGpuBuffers) when using CUDA acceleration.
    void *gpu_runtime{};
};
