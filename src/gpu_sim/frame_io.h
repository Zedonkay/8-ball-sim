#pragma once

#include "rigid_body.h"
#include "sim_state.h"

#include <cstdint>
#include <cstdio>


static constexpr uint32_t FRAME_MAGIC = 0x464D4932u; // "SIM2"

inline void
write_frame(FILE *f, const SimState &s, const RigidDie &die, int step, float t)
{
    const uint32_t magic   = FRAME_MAGIC;
    const int32_t  n_fluid = s.n_fluid;
    const int32_t  n_total = s.n_total;

    std::fwrite(&magic, sizeof(magic), 1, f);
    std::fwrite(&step, sizeof(step), 1, f);
    std::fwrite(&t, sizeof(t), 1, f);
    std::fwrite(&n_fluid, sizeof(n_fluid), 1, f);
    std::fwrite(&n_total, sizeof(n_total), 1, f);

    std::fwrite(s.pos_x, sizeof(float), n_total, f);
    std::fwrite(s.pos_y, sizeof(float), n_total, f);
    std::fwrite(s.pos_z, sizeof(float), n_total, f);
    std::fwrite(s.density, sizeof(float), n_fluid, f);
    std::fwrite(&die.pos.x, sizeof(float), 1, f);
    std::fwrite(&die.pos.y, sizeof(float), 1, f);
    std::fwrite(&die.pos.z, sizeof(float), 1, f);
    std::fwrite(&die.orient.w, sizeof(float), 1, f);
    std::fwrite(&die.orient.x, sizeof(float), 1, f);
    std::fwrite(&die.orient.y, sizeof(float), 1, f);
    std::fwrite(&die.orient.z, sizeof(float), 1, f);
    std::fwrite(&die.half_extents.x, sizeof(float), 1, f);
    std::fwrite(&die.half_extents.y, sizeof(float), 1, f);
    std::fwrite(&die.half_extents.z, sizeof(float), 1, f);

    std::fflush(f);
}
