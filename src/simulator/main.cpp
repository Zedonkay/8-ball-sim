#include <climits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "constants.h"
#include "frame_io.h"
#include "input_controls.h"
#include "initialization.h"
#include "rigid_body.h"
#include "sim.h"

static SimState s;
static RigidDie die;

const char *dt_limiter_name(DtLimiter limiter);

void print_usage(const char *prog)
{
    std::printf("Usage: %s [--verbose] [--no-output] [--keys-file path] "
                "[--high-agitation-profile] "
                "[--parity-debug-log path] [--parity-ghost-eps value] "
                "[--stable-live-profile] [--max-steps N] "
                "[--timing-output path] [output_file]\n",
                prog);
}

bool write_timing_json(const char *path,
                       double      initialization_s,
                       double      computation_s,
                       double      total_wall_s)
{
    std::ofstream out(path);
    if(!out)
    {
        std::fprintf(stderr, "[error] could not open timing output '%s'\n", path);
        return false;
    }
    out.setf(std::ios::fixed);
    out.precision(9);
    out << "{\n"
        << "  \"initialization_s\": " << initialization_s << ",\n"
        << "  \"computation_s\": " << computation_s << ",\n"
        << "  \"total_wall_s\": " << total_wall_s << "\n"
        << "}\n";
    return true;
}

struct ParityDebugOptions
{
    bool        enabled        = false;
    const char *path           = nullptr;
    float       ghost_eps      = 1e-4f;
    FILE       *stream         = nullptr;
};

struct GhostParityStats
{
    int   count                 = 0;
    int   near_threshold_count  = 0;
    float min_abs_wall_delta    = 0.0f;
    float max_abs_wall_delta    = 0.0f;
};

std::uint64_t hash(std::uint64_t h, std::uint32_t word)
{
    constexpr std::uint64_t kPrime = 1099511628211ull;
    h ^= static_cast<std::uint64_t>(word);
    h *= kPrime;
    return h;
}

std::uint32_t bit_cast_u32(float v)
{
    std::uint32_t out = 0;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}

std::uint64_t compute_fluid_state_hash(const SimState &state)
{
    std::uint64_t h = 1469598103934665603ull;
    for(int i = 0; i < state.n_fluid; ++i)
    {
        h = hash(h, bit_cast_u32(state.pos_x[i]));
        h = hash(h, bit_cast_u32(state.pos_y[i]));
        h = hash(h, bit_cast_u32(state.pos_z[i]));
        h = hash(h, bit_cast_u32(state.density[i]));
    }
    return h;
}

GhostParityStats compute_ghost_parity_stats(const SimState &state,
                                            float           threshold_eps)
{
    GhostParityStats out{};
    out.min_abs_wall_delta = 1e30f;
    for(int i = 0; i < state.n_fluid; ++i)
    {
        const float px    = state.pos_x[i];
        const float py    = state.pos_y[i];
        const float pz    = state.pos_z[i];
        const float r     = std::sqrt(px * px + py * py + pz * pz);
        const float delta = (SPHERE_R - r) - H;
        const float abs_d = std::fabs(delta);
        if(abs_d < out.min_abs_wall_delta)
            out.min_abs_wall_delta = abs_d;
        if(abs_d > out.max_abs_wall_delta)
            out.max_abs_wall_delta = abs_d;
        if(abs_d <= threshold_eps)
            ++out.near_threshold_count;
    }
    if(state.n_fluid <= 0)
    {
        out.min_abs_wall_delta = 0.0f;
        out.max_abs_wall_delta = 0.0f;
    }
    out.count = state.n_total - state.n_fluid;
    return out;
}

void emit_parity_log_line(FILE                        *stream,
                          int                          step,
                          float                        t,
                          const GhostParityStats      &ghost,
                          const RigidClampDiagnostics &rigid_clamp,
                          const DtDiagnostics         &dt_diagnostic,
                          std::uint64_t                fluid_hash)
{
    std::fprintf(stream,
                 "parity step=%d t=%.9f ghosts=%d near_ghost=%d "
                 "ghost_abs_delta_min=%.9e ghost_abs_delta_max=%.9e "
                 "die_force_pre=(%.9e,%.9e,%.9e) die_force_post=(%.9e,%.9e,%.9e) "
                 "die_torque_pre=(%.9e,%.9e,%.9e) die_torque_post=(%.9e,%.9e,%.9e) "
                 "die_force_norm_pre=%.9e die_force_norm_post=%.9e "
                 "die_torque_norm_pre=%.9e die_torque_norm_post=%.9e "
                 "dt=%.9e dt_adv=%.9e dt_ac=%.9e dt_force=%.9e dt_visc=%.9e "
                 "max_speed=%.9e max_accel=%.9e dt_limiter=%s fluid_hash=%016llx\n",
                 step,
                 t,
                 ghost.count,
                 ghost.near_threshold_count,
                 ghost.min_abs_wall_delta,
                 ghost.max_abs_wall_delta,
                 rigid_clamp.force_pre.x,
                 rigid_clamp.force_pre.y,
                 rigid_clamp.force_pre.z,
                 rigid_clamp.force_post.x,
                 rigid_clamp.force_post.y,
                 rigid_clamp.force_post.z,
                 rigid_clamp.torque_pre.x,
                 rigid_clamp.torque_pre.y,
                 rigid_clamp.torque_pre.z,
                 rigid_clamp.torque_post.x,
                 rigid_clamp.torque_post.y,
                 rigid_clamp.torque_post.z,
                 rigid_clamp.force_pre_norm,
                 rigid_clamp.force_post_norm,
                 rigid_clamp.torque_pre_norm,
                 rigid_clamp.torque_post_norm,
                 dt_diagnostic.dt,
                 dt_diagnostic.dt_advective,
                 dt_diagnostic.dt_acoustic,
                 dt_diagnostic.dt_force,
                 dt_diagnostic.dt_viscous,
                 dt_diagnostic.max_fluid_speed,
                 dt_diagnostic.max_fluid_accel,
                 dt_limiter_name(dt_diagnostic.limiter),
                 static_cast<unsigned long long>(fluid_hash));
}

const char *dt_limiter_name(DtLimiter limiter)
{
    switch(limiter)
    {
    case DtLimiter::Advective:
        return "advective";
    case DtLimiter::Acoustic:
        return "acoustic";
    case DtLimiter::Force:
        return "force";
    case DtLimiter::Viscous:
        return "viscous";
    default:
        return "unknown";
    }
}

int main(int argc, char *argv[])
{
    using Clock = std::chrono::steady_clock;
    const Clock::time_point run_t0 = Clock::now();

    bool        verbose                = false;
    bool        no_output              = false;
    bool        high_agitation_profile = false;
    bool        stable_live_profile    = false;
    ParityDebugOptions parity_debug{};
    const char *out_path               = "sim.bin";
    const char *keys_path              = nullptr;
    const char *timing_output_path     = nullptr;
    int         max_steps              = -1;
    for(int i = 1; i < argc;)
    {
        const char *arg = argv[i];
        if(std::strcmp(arg, "--verbose") == 0 || std::strcmp(arg, "-v") == 0)
        {
            verbose = true;
            ++i;
        }
        else if(std::strcmp(arg, "--no-output") == 0)
        {
            no_output = true;
            ++i;
        }
        else if(std::strcmp(arg, "--keys-file") == 0)
        {
            if(i + 1 >= argc)
            {
                print_usage(argv[0]);
                return 1;
            }
            keys_path = argv[i + 1];
            i += 2;
        }
        else if(std::strcmp(arg, "--high-agitation-profile") == 0)
        {
            high_agitation_profile = true;
            ++i;
        }
        else if(std::strcmp(arg, "--stable-live-profile") == 0)
        {
            stable_live_profile = true;
            ++i;
        }
        else if(std::strcmp(arg, "--max-steps") == 0)
        {
            if(i + 1 >= argc)
            {
                print_usage(argv[0]);
                return 1;
            }
            char      *end = nullptr;
            const long v   = std::strtol(argv[i + 1], &end, 10);
            if(end == argv[i + 1] || *end != '\0' || v <= 0 ||
               v > static_cast<long>(INT_MAX))
            {
                std::printf(
                    "[error] --max-steps requires a positive 32-bit integer\n");
                return 1;
            }
            max_steps = static_cast<int>(v);
            i += 2;
        }
        else if(std::strcmp(arg, "--timing-output") == 0)
        {
            if(i + 1 >= argc)
            {
                print_usage(argv[0]);
                return 1;
            }
            timing_output_path = argv[i + 1];
            i += 2;
        }
        else if(std::strcmp(arg, "--parity-debug-log") == 0)
        {
            if(i + 1 >= argc)
            {
                print_usage(argv[0]);
                return 1;
            }
            parity_debug.enabled = true;
            parity_debug.path    = argv[i + 1];
            i += 2;
        }
        else if(std::strcmp(arg, "--parity-ghost-eps") == 0)
        {
            if(i + 1 >= argc)
            {
                print_usage(argv[0]);
                return 1;
            }
            parity_debug.ghost_eps = std::fmax(0.0f, std::atof(argv[i + 1]));
            i += 2;
        }
        else if(arg[0] == '-')
        {
            print_usage(argv[0]);
            return 1;
        }
        else
        {
            out_path = arg;
            ++i;
        }
    }

    SIM_NO_OUTPUT = no_output;
    if(parity_debug.enabled)
    {
        parity_debug.stream = std::fopen(parity_debug.path, "w");
        if(!parity_debug.stream)
        {
            std::printf("[error] could not open parity log '%s' for writing\n",
                        parity_debug.path);
            return 1;
        }
    }
    ShakeInputState shake;
    shake.output_enabled = !no_output;
    if(stable_live_profile)
    {
        enable_stable_live_profile(shake);
        if(verbose && !no_output)
        {
            std::printf("[profile] stable-live enabled\n");
        }
    }
    if(high_agitation_profile)
    {
        enable_high_agitation_profile(shake);
        if(verbose && !no_output)
        {
            std::printf("[profile] high-agitation enabled\n");
        }
    }
    ScriptedKeyEvents scripted_keys;
    if(keys_path && !scripted_keys.load(keys_path, verbose, !no_output, true))
        return 1;

    FILE *out = nullptr;
    if(!no_output)
    {
        out = std::fopen(out_path, "wb");
        if(!out)
        {
            std::printf("[error] could not open '%s' for writing\n", out_path);
            return 1;
        }
    }

    poisson_disk_sample_sphere(s);
    die = init_rigid_die();
    if(verbose && !no_output)
    {
        std::printf(
            "[init] %d fluid particles placed inside sphere (r=%.3f m)\n",
            s.n_fluid,
            SPHERE_R);
    }
    KeyboardControl keyboard;
    bool            keyboard_active = false;
    if(keyboard.init(&shake, verbose))
    {
        keyboard_active = true;
        if(!no_output)
        {
            print_keyboard_controls();
        }
    }
    else
    {
        if(verbose && !no_output)
            std::printf("[controls] keyboard shaking disabled (stdin is not "
                        "interactive)\n");
    }

    float t = 0.0f;
    int   step = 0;
    Vec3  prev_accel{};
    Vec3  prev_torque{};
    if(out)
        write_frame(out, s, die, step, t);

    const bool cap_steps = max_steps > 0;
    const Clock::time_point compute_t0 = Clock::now();
    while(cap_steps ? (step < max_steps) : (t < TOTAL_TIME))
    {
        scripted_keys.apply_through_time(t, shake);
        keyboard.poll();
        if(shake.pending_clear_control)
        {
            prev_accel                  = shake.accel;
            prev_torque                 = shake.torque;
            shake.pending_clear_control = false;
        }
        else
        {
            const Vec3 accel_delta  = shake.accel - prev_accel;
            const Vec3 torque_delta = shake.torque - prev_torque;
            die.vel += DIE_KEY_VELOCITY_GAIN_RUNTIME * accel_delta;
            die.omega += DIE_KEY_OMEGA_GAIN_RUNTIME * torque_delta;
            die.vel   = clamp_vec3_length(die.vel, RIGID_MAX_SPEED_RUNTIME);
            die.omega = clamp_vec3_length(die.omega, RIGID_MAX_OMEGA_RUNTIME);
            prev_accel  = shake.accel;
            prev_torque = shake.torque;
        }
        Vec3 external_accel  = shake.accel;
        Vec3 external_torque = shake.torque;
        inject_ghost_particles(s, die);
        DtDiagnostics dt_diagnostic{};
        RigidClampDiagnostics rigid_clamp_diagnostic{};
        float dt = step_sim(s,
                            die,
                            external_accel,
                            external_torque,
                            &dt_diagnostic,
                            &rigid_clamp_diagnostic);

        t += dt;
        step += 1;
        if(parity_debug.enabled && parity_debug.stream)
        {
            const GhostParityStats ghost_stats =
                compute_ghost_parity_stats(s, parity_debug.ghost_eps);
            const std::uint64_t fluid_hash = compute_fluid_state_hash(s);
            emit_parity_log_line(parity_debug.stream,
                                 step,
                                 t,
                                 ghost_stats,
                                 rigid_clamp_diagnostic,
                                 dt_diagnostic,
                                 fluid_hash);
        }

        if(verbose && !no_output && step % LOG_EVERY == 0)
        {
            std::printf(
                "[step %5d] t=%.5f s  dt=%.2e s  ghosts=%d  fluid|v|_max=%.4f  "
                "die_accel=(%+.2f,%+.2f,%+.2f)  die_torque=(%+.3f,%+.3f,%+.3f)  "
                "dt_limit=%s (adv=%.2e ac=%.2e force=%.2e visc=%.2e)\n",
                step,
                t,
                dt,
                s.n_total - s.n_fluid,
                dt_diagnostic.max_fluid_speed,
                shake.accel.x,
                shake.accel.y,
                shake.accel.z,
                shake.torque.x,
                shake.torque.y,
                shake.torque.z,
                dt_limiter_name(dt_diagnostic.limiter),
                dt_diagnostic.dt_advective,
                dt_diagnostic.dt_acoustic,
                dt_diagnostic.dt_force,
                dt_diagnostic.dt_viscous);
        }
        else if(!no_output && keyboard_active && step % LOG_EVERY == 0)
        {
            std::printf(
                "[input] accel=(%+.2f,%+.2f,%+.2f) torque=(%+.3f,%+.3f,%+.3f) "
                "steps(accel=%.2f torque=%.3f)\n",
                external_accel.x,
                external_accel.y,
                external_accel.z,
                external_torque.x,
                external_torque.y,
                external_torque.z,
                shake.accel_step,
                shake.torque_step);
            std::fflush(stdout);
        }

        if(out)
            write_frame(out, s, die, step, t);
    }

    if(out)
        std::fclose(out);
    const Clock::time_point run_t1 = Clock::now();
    const double initialization_s =
        std::chrono::duration<double>(compute_t0 - run_t0).count();
    const double total_wall_s =
        std::chrono::duration<double>(run_t1 - run_t0).count();
    const double computation_s = total_wall_s - initialization_s;
    if(timing_output_path &&
       !write_timing_json(
           timing_output_path, initialization_s, computation_s, total_wall_s))
    {
        return 1;
    }
    if(parity_debug.stream)
        std::fclose(parity_debug.stream);
    if(verbose && !no_output)
    {
        std::printf("[done] simulated %.3f s in %d steps\n", t, step);
        std::printf("[dump] frames written to '%s'\n", out_path);
    }
    return 0;
}
