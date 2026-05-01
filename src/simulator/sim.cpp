#include "sim.h"

#include "box_sdf.h"
#include "constants.h"
#include "kernels.h"
#include "rigid_body.h"
#include "sim_state.h"
#include "spatial_hash.h"

#include <cstring>

void compute_density(SimState &s)
{
    for(int i = 0; i < s.n_fluid; ++i)
    {
        float rho = s.mass[i] * W_poly6(0.f);
        int   nc  = s.neighbor_count[i];
        for(int kk = 0; kk < nc; ++kk)
        {
            int   j    = s.neighbor_list[i * MAX_NEIGHBORS + kk];
            float dx   = s.pos_x[i] - s.pos_x[j];
            float dy   = s.pos_y[i] - s.pos_y[j];
            float dz   = s.pos_z[i] - s.pos_z[j];
            float r_sq = dx * dx + dy * dy + dz * dz;
            rho += s.mass[j] * W_poly6(r_sq);
        }
        s.density[i] = std::fmax(rho, MIN_RHO);
    }
}

void compute_pressure(SimState &s)
{
    for(int i = 0; i < s.n_fluid; ++i)
    {
        float ratio   = s.density[i] / s.params.rest_density;
        float r3      = ratio * ratio * ratio;
        float r7      = r3 * r3 * ratio;
        float p       = s.params.stiffness * (r7 - 1.0f);
        s.pressure[i] = std::fmax(0.0f, p);
    }
}

void compute_forces(SimState &s)
{
    const int v_read = s.ping;
    for(int i = 0; i < s.n_fluid; ++i)
    {
        float fp_x = 0.0f, fp_y = 0.0f, fp_z = 0.0f;
        float fv_x = 0.0f, fv_y = 0.0f, fv_z = 0.0f;
        float xs_x = 0.0f, xs_y = 0.0f, xs_z = 0.0f;

        Vec3  pos_i = {s.pos_x[i], s.pos_y[i], s.pos_z[i]};
        float vx_i  = s.vel_x[i][v_read];
        float vy_i  = s.vel_y[i][v_read];
        float vz_i  = s.vel_z[i][v_read];
        float rho_i = std::fmax(s.density[i], MIN_RHO);
        float p_i   = s.pressure[i];
        float m_i   = s.mass[i];

        int nc = s.neighbor_count[i];
        for(int kk = 0; kk < nc; ++kk)
        {
            int   j    = s.neighbor_list[i * MAX_NEIGHBORS + kk];
            Vec3  dpos = pos_i - Vec3{s.pos_x[j], s.pos_y[j], s.pos_z[j]};
            float r_sq = dpos.length_sq();
            if(r_sq < MIN_R2 || r_sq >= H * H)
                continue;

            float r     = std::sqrt(r_sq);
            float rho_j = std::fmax(s.density[j], MIN_RHO);
            float p_j   = s.pressure[j];
            float m_j   = s.mass[j];

            Vec3  grad_w  = gradW_spiky(dpos, r);
            float sym_p   = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
            float coeff_p = m_i * m_j * sym_p;
            fp_x -= coeff_p * grad_w.x;
            fp_y -= coeff_p * grad_w.y;
            fp_z -= coeff_p * grad_w.z;

            float lap_w   = lapW_viscosity(r);
            float coeff_v = m_i * s.params.viscosity * m_j / (rho_i * rho_j) * lap_w;
            fv_x += coeff_v * (s.vel_x[j][v_read] - vx_i);
            fv_y += coeff_v * (s.vel_y[j][v_read] - vy_i);
            fv_z += coeff_v * (s.vel_z[j][v_read] - vz_i);

            if(!s.is_ghost[j])
            {
                float w_p6    = W_poly6(r_sq);
                float w_coeff = (m_j / rho_j) * w_p6;
                xs_x += w_coeff * (s.vel_x[j][v_read] - vx_i);
                xs_y += w_coeff * (s.vel_y[j][v_read] - vy_i);
                xs_z += w_coeff * (s.vel_z[j][v_read] - vz_i);
            }
        }

        s.force_x[i]    = fp_x + fv_x + m_i * GRAVITY.x;
        s.force_y[i]    = fp_y + fv_y + m_i * GRAVITY.y;
        s.force_z[i]    = fp_z + fv_z + m_i * GRAVITY.z;
        s.vel_xsph_x[i] = XSPH_EPS * xs_x;
        s.vel_xsph_y[i] = XSPH_EPS * xs_y;
        s.vel_xsph_z[i] = XSPH_EPS * xs_z;
    }
}

void handle_die_coupling(SimState &s, RigidDie &die)
{
    die.force_accum  = {0.0f, 0.0f, 0.0f};
    die.torque_accum = {0.0f, 0.0f, 0.0f};

    const int v_read     = s.ping;
    Quat      inv_orient = die.orient.conjugate();

    for(int i = 0; i < s.n_fluid; ++i)
    {
        Vec3      p         = {s.pos_x[i], s.pos_y[i], s.pos_z[i]};
        Vec3      local_pos = inv_orient.rotate(p - die.pos);
        die_dimensions sdf       = sdf_box(local_pos, die.half_extents);
        if(sdf.dist >= DIE_CONTACT_MARGIN)
            continue;

        Vec3        world_normal = die.orient.rotate(sdf.normal);
        const float penetration =
            std::fmax(0.0f, DIE_CONTACT_MARGIN - sdf.dist);
        Vec3 fc = CONTACT_K * penetration * world_normal;

        Vec3 r_world = p - die.pos;
        Vec3 v_surf  = die.vel + die.omega.cross(r_world);
        Vec3 v_fluid = {
            s.vel_x[i][v_read], s.vel_y[i][v_read], s.vel_z[i][v_read]};
        Vec3        v_rel  = v_fluid - v_surf;
        const float vn_rel = v_rel.dot(world_normal);
        if(vn_rel < 0.0f)
        {
            fc += (-DIE_CONTACT_DAMPING * vn_rel) * world_normal;
        }
        float speed_sq = v_rel.length_sq();
        Vec3  fd{0.0f, 0.0f, 0.0f};
        if(speed_sq > 1e-8f)
        {
            float drag_mag = 0.5f * s.density[i] * speed_sq *
                             s.params.drag_coeff * PARTICLE_AREA;
            drag_mag       = std::fmin(drag_mag, 0.5f * RIGID_MAX_FORCE);
            fd             = drag_mag * v_rel.normalized();
        }

        s.force_x[i] += fc.x - fd.x;
        s.force_y[i] += fc.y - fd.y;
        s.force_z[i] += fc.z - fd.z;

        Vec3 react = {-fc.x + fd.x, -fc.y + fd.y, -fc.z + fd.z};
        die.force_accum += react;
        die.torque_accum += r_world.cross(react);
    }
}

DtDiagnostics get_dt(SimState &s)
{
    const int v_read    = s.ping;
    float     max_speed = 0.0f;
    float     max_accel = 0.0f;

    for(int i = 0; i < s.n_fluid; ++i)
    {
        Vec3 v = {s.vel_x[i][v_read], s.vel_y[i][v_read], s.vel_z[i][v_read]};
        max_speed = std::fmax(max_speed, v.length());

        const float inv_mass = 1.0f / std::fmax(s.mass[i], MIN_RHO);
        Vec3        a        = {s.force_x[i] * inv_mass,
                                s.force_y[i] * inv_mass,
                                s.force_z[i] * inv_mass};
        max_accel            = std::fmax(max_accel, a.length());
    }

    const float eps  = 1e-6f;
    const float rho0 = std::fmax(s.params.rest_density, MIN_RHO);
    const float nu   = s.params.viscosity / rho0;
    const float c0_from_eos =
        std::sqrt(static_cast<float>(GAMMA) * s.params.stiffness / rho0);
    const float c_s =
        std::fmax(CFL_SOUND_SPEED_VEL_SCALE * max_speed, c0_from_eos);

    float         dt_advective = CFL_LAMBDA * H / (max_speed + eps);
    float         dt_acoustic  = CFL_LAMBDA * H / (c_s + eps);
    float         dt_force     = CFL_LAMBDA * std::sqrt(H / (max_accel + eps));
    float         dt_viscous   = CFL_LAMBDA * H * H / (nu + eps);
    DtDiagnostics out{};
    out.dt_advective = dt_advective;
    out.dt_acoustic  = dt_acoustic;
    out.dt_force     = dt_force;
    out.dt_viscous   = dt_viscous;

    out.dt      = out.dt_advective;
    out.limiter = DtLimiter::Advective;
    if(out.dt_acoustic < out.dt)
    {
        out.dt      = out.dt_acoustic;
        out.limiter = DtLimiter::Acoustic;
    }
    if(out.dt_force < out.dt)
    {
        out.dt      = out.dt_force;
        out.limiter = DtLimiter::Force;
    }
    if(out.dt_viscous < out.dt)
    {
        out.dt      = out.dt_viscous;
        out.limiter = DtLimiter::Viscous;
    }

    out.dt              = std::fmax(DT_MIN, std::fmin(out.dt, DT_MAX));
    out.max_fluid_speed = max_speed;
    out.max_fluid_accel = max_accel;
    return out;
}

void integrate(SimState &s, float dt)
{
    const int v_read  = s.ping;
    const int v_write = s.pong;

    for(int i = 0; i < s.n_fluid; ++i)
    {
        const float inv_mass = 1.0f / std::fmax(s.mass[i], MIN_RHO);
        Vec3        a        = {s.force_x[i] * inv_mass,
                                s.force_y[i] * inv_mass,
                                s.force_z[i] * inv_mass};
        Vec3        v_old    = {
            s.vel_x[i][v_read], s.vel_y[i][v_read], s.vel_z[i][v_read]};
        Vec3 v_new = v_old + dt * a;
        Vec3 v_adv = {
            v_new.x + s.vel_xsph_x[i],
            v_new.y + s.vel_xsph_y[i],
            v_new.z + s.vel_xsph_z[i],
        };

        s.vel_x[i][v_write] = v_new.x;
        s.vel_y[i][v_write] = v_new.y;
        s.vel_z[i][v_write] = v_new.z;
        s.pos_x[i] += v_adv.x * dt;
        s.pos_y[i] += v_adv.y * dt;
        s.pos_z[i] += v_adv.z * dt;
    }

    std::memset(s.force_x, 0, s.n_fluid * sizeof(float));
    std::memset(s.force_y, 0, s.n_fluid * sizeof(float));
    std::memset(s.force_z, 0, s.n_fluid * sizeof(float));

    s.swap_buffers();
}

void handle_boundary(SimState &s)
{
    const float boundary_r         = SPHERE_R - PARTICLE_R;
    const float boundary_r2        = boundary_r * boundary_r;
    const float restitution        = s.params.wall_restitution;
    const float tangential_damping = 1.0f - s.params.friction;
    const int   v_active           = s.ping;

    for(int i = 0; i < s.n_fluid; ++i)
    {
        Vec3 pos = {s.pos_x[i], s.pos_y[i], s.pos_z[i]};
        if(pos.length_sq() >= boundary_r2)
        {
            Vec3 n = pos.normalized();
            pos    = boundary_r * n;

            Vec3 v = {
                s.vel_x[i][v_active],
                s.vel_y[i][v_active],
                s.vel_z[i][v_active],
            };

            float vn_scalar = v.dot(n);
            if(vn_scalar > 0.0f)
            {
                Vec3 v_n = vn_scalar * n;
                Vec3 v_t = v - v_n;
                v        = (-restitution) * v_n + tangential_damping * v_t;
            }

            s.pos_x[i]           = pos.x;
            s.pos_y[i]           = pos.y;
            s.pos_z[i]           = pos.z;
            s.vel_x[i][v_active] = v.x;
            s.vel_y[i][v_active] = v.y;
            s.vel_z[i][v_active] = v.z;
        }
    }
}

float step_sim(SimState      &s,
               RigidDie      &die,
               const Vec3    &external_accel,
               const Vec3    &external_torque,
               DtDiagnostics *dt_diagnostic,
               RigidClampDiagnostics *rigid_clamp_diagnostic)
{
    build_spatial_hash(s);
    build_neighbor_lists(s);
    compute_density(s);
    compute_pressure(s);
    compute_forces(s);
    handle_die_coupling(s, die);
    DtDiagnostics diag = get_dt(s);
    integrate(s, diag.dt);
    handle_boundary(s);
    integrate_rigid_body(die,
                         diag.dt,
                         external_accel,
                         external_torque,
                         rigid_clamp_diagnostic);
    clamp_die_inside_sphere(die);
    if(dt_diagnostic)
        *dt_diagnostic = diag;
    return diag.dt;
}
