#pragma once
#include "sim_state.h"

inline int cell_coord(float v)
{
    return static_cast<int>(std::floor(v / CELL_SIZE));
}

inline int spatial_hash(int ix, int iy, int iz)
{
    return std::abs(ix * 73856093 ^ iy * 19349663 ^ iz * 83492791) % TABLE_SIZE;
}

void build_spatial_hash(SimState &s);
void build_neighbor_lists(SimState &s);
