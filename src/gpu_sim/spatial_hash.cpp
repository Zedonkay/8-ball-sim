#include "spatial_hash.h"

#include <algorithm>
#include <cstring>
#include <numeric>

void build_spatial_hash(SimState &s)
{
    const int n = s.n_total;

    for(int i = 0; i < n; ++i)
    {
        int ix         = cell_coord(s.pos_x[i]);
        int iy         = cell_coord(s.pos_y[i]);
        int iz         = cell_coord(s.pos_z[i]);
        s.cell_ix[i]   = ix;
        s.cell_iy[i]   = iy;
        s.cell_iz[i]   = iz;
        s.hash_key[i]  = spatial_hash(ix, iy, iz);
        s.sorted_id[i] = i;
    }

    std::sort(s.sorted_id,
              s.sorted_id + n,
              [&](int a, int b) { return s.hash_key[a] < s.hash_key[b]; });

    std::memset(s.cell_count, 0, TABLE_SIZE * sizeof(int));

    for(int rank = 0; rank < n; ++rank)
    {
        int k = s.hash_key[s.sorted_id[rank]];
        s.cell_count[k]++;
    }

    s.cell_start[0] = 0;
    for(int k = 1; k < TABLE_SIZE; ++k)
    {
        s.cell_start[k] = s.cell_start[k - 1] + s.cell_count[k - 1];
    }
}

void build_neighbor_lists(SimState &s)
{
    const float h_sq = H * H;

    for(int i = 0; i < s.n_fluid; ++i)
    {
        int count = 0;

        int ix = cell_coord(s.pos_x[i]);
        int iy = cell_coord(s.pos_y[i]);
        int iz = cell_coord(s.pos_z[i]);

        // Scan 3×3×3 = 27 neighboring cells
        for(int dx = -1; dx <= 1; ++dx)
        {
            for(int dy = -1; dy <= 1; ++dy)
            {
                for(int dz = -1; dz <= 1; ++dz)
                {
                    const int target_ix = ix + dx;
                    const int target_iy = iy + dy;
                    const int target_iz = iz + dz;
                    int       key   = spatial_hash(ix + dx, iy + dy, iz + dz);
                    int       start = s.cell_start[key];
                    int       cnt   = s.cell_count[key];
                    for(int off = 0; off < cnt; ++off)
                    {
                        int j = s.sorted_id[start + off];
                        if(j == i)
                            continue;
                        if(s.cell_ix[j] != target_ix ||
                           s.cell_iy[j] != target_iy ||
                           s.cell_iz[j] != target_iz)
                            continue;

                        float ddx  = s.pos_x[i] - s.pos_x[j];
                        float ddy  = s.pos_y[i] - s.pos_y[j];
                        float ddz  = s.pos_z[i] - s.pos_z[j];
                        float r_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                        if(r_sq < h_sq && count < MAX_NEIGHBORS)
                        {
                            s.neighbor_list[count * N_MAX + i] = j;
                            count++;
                        }
                    }
                }
            }
        }
        s.neighbor_count[i] = count;
    }
}
