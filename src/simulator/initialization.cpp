#include "initialization.h"

#include "constants.h"
#include "vec_math.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace
{
constexpr float DIE_INITIAL_CLEARANCE =
    DIE_HALF + DIE_CONTACT_MARGIN + PARTICLE_R;
constexpr float DIE_INITIAL_CAVITY_VOLUME =
    8.0f * DIE_INITIAL_CLEARANCE * DIE_INITIAL_CLEARANCE *
    DIE_INITIAL_CLEARANCE;

bool inside_initial_die_cavity(float x, float y, float z)
{
    return std::fabs(x) < DIE_INITIAL_CLEARANCE &&
           std::fabs(y) < DIE_INITIAL_CLEARANCE &&
           std::fabs(z) < DIE_INITIAL_CLEARANCE;
}
} // namespace

void poisson_disk_sample_sphere(SimState &s)
{

    const float inner_r    = SPHERE_R;
    const float inner_r_sq = inner_r * inner_r;
    const float sphere_volume =
        (4.0f / 3.0f) * PI * inner_r * inner_r * inner_r;
    const float effective_volume =
        std::fmax(1e-6f, sphere_volume - DIE_INITIAL_CAVITY_VOLUME);
    const float target_count = static_cast<float>(std::max(1, N_PARTICLES));
    const float minimum_distance = std::cbrt(effective_volume / target_count);
    const float minimum_distance_sq = minimum_distance * minimum_distance;

    const int give_up_after = POISSON_GIVE_UP_AFTER;

    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> unit(0.f, 1.f);

    std::vector<Vec3> accepted;
    accepted.reserve(N_PARTICLES);

    std::vector<int> active;

    auto try_add = [&](float x, float y, float z) -> bool
    {
        float r_sq = x * x + y * y + z * z;
        if(r_sq >= inner_r_sq)
            return false;
        if(inside_initial_die_cavity(x, y, z))
            return false;
        for(const auto &p : accepted)
        {
            float dx = x - p.x, dy = y - p.y, dz = z - p.z;
            if(dx * dx + dy * dy + dz * dz < minimum_distance_sq)
                return false;
        }
        accepted.push_back({x, y, z});
        active.push_back(static_cast<int>(accepted.size()) - 1);
        return true;
    };

    try_add(DIE_INITIAL_CLEARANCE + minimum_distance, 0.f, 0.f);

    while(!active.empty() && static_cast<int>(accepted.size()) < N_PARTICLES)
    {
        std::uniform_int_distribution<int> seed_point(
            0, static_cast<int>(active.size()) - 1);
        int         idx  = seed_point(rng);
        const Vec3 &base = accepted[active[idx]];

        bool found_candidate = false;
        for(int k = 0; k < give_up_after; ++k)
        {
            float radius = minimum_distance * (1.f + unit(rng));
            float u      = 2.f * unit(rng) - 1.f;
            float phi    = 2.f * PI * unit(rng);

            float r_xy = std::sqrt(std::max(0.f, 1.f - u * u));

            float dx = radius * r_xy * std::cos(phi);
            float dy = radius * r_xy * std::sin(phi);
            float dz = radius * u;

            if(try_add(base.x + dx, base.y + dy, base.z + dz))
            {
                found_candidate = true;
            }
        }

        if(!found_candidate)
        {
            active[idx] = active.back();
            active.pop_back();
        }
    }
    int n = static_cast<int>(accepted.size());
    for(int i = 0; i < n; ++i)
    {
        s.pos_x[i] = accepted[i].x;
        s.pos_y[i] = accepted[i].y;
        s.pos_z[i] = accepted[i].z;

        s.vel_x[i][PING] = 0.f;
        s.vel_x[i][PONG] = 0.f;
        s.vel_y[i][PING] = 0.f;
        s.vel_y[i][PONG] = 0.f;
        s.vel_z[i][PING] = 0.f;
        s.vel_z[i][PONG] = 0.f;

        s.vel_xsph_x[i] = 0.f;
        s.vel_xsph_y[i] = 0.f;
        s.vel_xsph_z[i] = 0.f;

        s.force_x[i] = 0.f;
        s.force_y[i] = 0.f;
        s.force_z[i] = 0.f;

        s.mass[i]          = 0.0f;
        s.density[i]       = s.params.rest_density;
        s.pressure[i]      = 0.f;
        s.particle_type[i] = PARTICLE_TYPE_FLUID;
        s.is_ghost[i]      = false;
    }

    s.n_fluid = n;
    s.n_total = n;
    const float particle_mass =
        (n > 0)
            ? (s.params.rest_density * effective_volume / static_cast<float>(n))
            : PARTICLE_M;
    for(int i = 0; i < n; ++i)
    {
        s.mass[i] = particle_mass;
    }
}

void inject_ghost_particles(SimState &s, const RigidDie &die)
{
    int       g = s.n_fluid;
    const int p = s.ping;

    for(int i = 0; i < s.n_fluid; ++i)
    {
        float px = s.pos_x[i];
        float py = s.pos_y[i];
        float pz = s.pos_z[i];

        float r            = std::sqrt(px * px + py * py + pz * pz);
        float dist_to_wall = SPHERE_R - r;
        if(dist_to_wall >= H)
            continue;

        if(g >= N_MAX)
            break;

        float reciprocal_r = (r > 1e-12f) ? (1.f / r) : 0.f;
        float nx           = px * reciprocal_r;
        float ny           = py * reciprocal_r;
        float nz           = pz * reciprocal_r;
        if(r <= 1e-12f)
        {
            nx = 0.0f;
            ny = 0.0f;
            nz = 1.0f;
        }

        const float mirror_r = SPHERE_R + dist_to_wall;

        s.pos_x[g] = nx * mirror_r;
        s.pos_y[g] = ny * mirror_r;
        s.pos_z[g] = nz * mirror_r;

        s.vel_x[g][p]     = 0.0f;
        s.vel_y[g][p]     = 0.0f;
        s.vel_z[g][p]     = 0.0f;
        s.vel_x[g][1 - p] = 0.0f;
        s.vel_y[g][1 - p] = 0.0f;
        s.vel_z[g][1 - p] = 0.0f;

        s.mass[g]          = s.mass[i];
        s.density[g]       = s.params.rest_density;
        s.pressure[g]      = 0.0f;
        s.particle_type[g] = PARTICLE_TYPE_GHOST_WALL;
        s.is_ghost[g]      = true;

        g++;
    }

    const float ghost_mass =
        ((s.n_fluid > 0) ? s.mass[0] : PARTICLE_M) * DIE_GHOST_MASS_SCALE;
    for(int k = 0; k < die.n_ghost_offsets && g < N_MAX; ++k)
    {
        const Vec3 r_world = die.orient.rotate(die.ghost_offsets[k]);
        const Vec3 p_world = die.pos + r_world;
        const Vec3 v_surf  = die.vel + die.omega.cross(r_world);

        s.pos_x[g] = p_world.x;
        s.pos_y[g] = p_world.y;
        s.pos_z[g] = p_world.z;

        s.vel_x[g][PING] = v_surf.x;
        s.vel_y[g][PING] = v_surf.y;
        s.vel_z[g][PING] = v_surf.z;
        s.vel_x[g][PONG] = v_surf.x;
        s.vel_y[g][PONG] = v_surf.y;
        s.vel_z[g][PONG] = v_surf.z;

        s.vel_xsph_x[g] = 0.0f;
        s.vel_xsph_y[g] = 0.0f;
        s.vel_xsph_z[g] = 0.0f;
        s.force_x[g]    = 0.0f;
        s.force_y[g]    = 0.0f;
        s.force_z[g]    = 0.0f;

        s.mass[g]          = ghost_mass;
        s.density[g]       = s.params.rest_density;
        s.pressure[g]      = 0.0f;
        s.particle_type[g] = PARTICLE_TYPE_DIE_WALL;
        s.is_ghost[g]      = true;

        ++g;
    }

    s.n_total = g;
}
