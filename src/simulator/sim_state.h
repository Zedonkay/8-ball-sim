#pragma once

#include "constants.h"

#define PARTICLE_TYPE_FLUID      0
#define PARTICLE_TYPE_GHOST_WALL 1
#define PARTICLE_TYPE_DIE_WALL   2

struct SimState
{
    // ── Position
    float pos_x[N_MAX]{};
    float pos_y[N_MAX]{};
    float pos_z[N_MAX]{};

    // ── Velocity (double-buffered: read PING, write PONG, swap each step)
    float vel_x[N_MAX][2]{};
    float vel_y[N_MAX][2]{};
    float vel_z[N_MAX][2]{};

    // ── XSPH position-advection correction 
    float vel_xsph_x[N_MAX]{};
    float vel_xsph_y[N_MAX]{};
    float vel_xsph_z[N_MAX]{};

    // ── Accumulated force 
    float force_x[N_MAX]{};
    float force_y[N_MAX]{};
    float force_z[N_MAX]{};

    // ── Per-particle scalars
    float mass[N_MAX]{};
    float density[N_MAX]{};
    float pressure[N_MAX]{};

    // ── Particle classification
    int  particle_type[N_MAX]{};
    bool is_ghost[N_MAX]{};

    // ── Neighbor list (flat 2-D array)
    int neighbor_list[N_MAX * MAX_NEIGHBORS]{};
    int neighbor_count[N_MAX]{};

    // ── Spatial hash
    int hash_key[N_MAX]{}; // cell hash for particle i
    int cell_ix[N_MAX]{}; // grid cell x for particle i
    int cell_iy[N_MAX]{}; // grid cell y for particle i
    int cell_iz[N_MAX]{}; // grid cell z for particle i
    int sorted_id[N_MAX]{}; // particle indices sorted by hash_key
    int cell_start[TABLE_SIZE]{}; // first index in sorted_id for each bucket
    int cell_count[TABLE_SIZE]{}; // number of particles per hash bucket

    // ── Runtime counters
    int n_fluid{}; // fluid-only particles 
    int n_total{}; // fluid + ghost;

    // ── Ping-pong buffer selector
    int ping{PING};
    int pong{PONG};

    void swap_buffers()
    {
        int tmp = ping;
        ping    = pong;
        pong    = tmp;
    }

    // ── parameters 
    Params params{};
};
