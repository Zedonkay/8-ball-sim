#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include "constants.h"
#include "frame_io.h"
#include "../common/input_controls.h"
#include "initialization.h"
#include "rigid_body.h"
#include "sim.h"
#include "sim_cuda.h"

static SimState s;
static RigidDie die;

void print_usage(const char *prog)
{
    std::printf("Usage: %s [--verbose] [--no-output] [--keys-file path] "
                "[--high-agitation-profile] "
                "[--stable-live-profile] "
                "[--gpu-strict-sync] [--fused-forces] "
                "[--neighbor-rebuild-every K] [--output-every K] "
                "[--realtime-output] "
                "[--parity-debug-log path] [--parity-ghost-eps value] "
                "[--benchmark-timings] [--max-steps N] [output_file]\n",
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
    bool        enabled   = false;
    const char *path      = nullptr;
    float       ghost_eps = 1e-4f;
    FILE       *stream    = nullptr;
};

struct GhostParityStats
{
    int   count                = 0;
    int   near_threshold_count = 0;
    float min_abs_wall_delta   = 0.0f;
    float max_abs_wall_delta   = 0.0f;
};

std::uint64_t fnv1a64_mix(std::uint64_t h, std::uint32_t word)
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
        h = fnv1a64_mix(h, bit_cast_u32(state.pos_x[i]));
        h = fnv1a64_mix(h, bit_cast_u32(state.pos_y[i]));
        h = fnv1a64_mix(h, bit_cast_u32(state.pos_z[i]));
        h = fnv1a64_mix(h, bit_cast_u32(state.density[i]));
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

const char *dt_limiter_name_gpu(int limiter)
{
    switch(limiter)
    {
    case 0:
        return "advective";
    case 1:
        return "acoustic";
    case 2:
        return "force";
    case 3:
        return "viscous";
    default:
        return "unknown";
    }
}

float vec_norm3(const float v[3])
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void emit_parity_log_line(FILE                   *stream,
                          int                     step,
                          float                   t,
                          const GhostParityStats &ghost,
                          const SimGpuParityDebug &dbg,
                          std::uint64_t           fluid_hash)
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
                 dbg.die_force_pre[0],
                 dbg.die_force_pre[1],
                 dbg.die_force_pre[2],
                 dbg.die_force_post[0],
                 dbg.die_force_post[1],
                 dbg.die_force_post[2],
                 dbg.die_torque_pre[0],
                 dbg.die_torque_pre[1],
                 dbg.die_torque_pre[2],
                 dbg.die_torque_post[0],
                 dbg.die_torque_post[1],
                 dbg.die_torque_post[2],
                 vec_norm3(dbg.die_force_pre),
                 vec_norm3(dbg.die_force_post),
                 vec_norm3(dbg.die_torque_pre),
                 vec_norm3(dbg.die_torque_post),
                 dbg.dt,
                 dbg.dt_advective,
                 dbg.dt_acoustic,
                 dbg.dt_force,
                 dbg.dt_viscous,
                 dbg.max_speed,
                 dbg.max_accel,
                 dt_limiter_name_gpu(dbg.dt_limiter),
                 static_cast<unsigned long long>(fluid_hash));
}


int main(int argc, char *argv[])
{
    using Clock = std::chrono::steady_clock;
    const Clock::time_point run_t0 = Clock::now();

    bool        verbose                = false;
    bool        no_output              = false;
    bool        high_agitation_profile = false;
    bool        stable_live_profile    = false;
    bool        benchmark_timings      = false;
    bool        realtime_output        = false;
    ParityDebugOptions parity_debug{};
    const char *out_path               = "sim.bin";
    const char *keys_path              = nullptr;
    const char *timing_output_path     = nullptr;
    int         max_steps              = -1;
    int         output_every           = 1;
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
        else if(std::strcmp(arg, "--benchmark-timings") == 0)
        {
            benchmark_timings = true;
            ++i;
        }
        else if(std::strcmp(arg, "--realtime-output") == 0)
        {
            realtime_output = true;
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
        else if(std::strcmp(arg, "--gpu-strict-sync") == 0)
        {
            SIM_GPU_STRICT_SYNC = true;
            ++i;
        }
        else if(std::strcmp(arg, "--fused-forces") == 0)
        {
            SIM_GPU_FUSED_FORCES = true;
            ++i;
        }
        else if(std::strcmp(arg, "--neighbor-rebuild-every") == 0)
        {
            if(i + 1 >= argc)
            {
                print_usage(argv[0]);
                return 1;
            }
            SIM_NEIGHBOR_REBUILD_EVERY = std::max(1, std::atoi(argv[i + 1]));
            i += 2;
        }
        else if(std::strcmp(arg, "--output-every") == 0)
        {
            if(i + 1 >= argc)
            {
                print_usage(argv[0]);
                return 1;
            }
            output_every = std::max(1, std::atoi(argv[i + 1]));
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
    SIM_VERBOSE   = verbose;
    const auto benchmark_start = std::chrono::steady_clock::now();
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
    shake.output_enabled            = !no_output;
    shake.announce_requires_verbose = true;
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
    if(keys_path && !scripted_keys.load(keys_path, verbose, !no_output))
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
    if(!sim_gpu_ensure_initialized(s))
        return 1;
    if(!sim_gpu_upload_die(s, die))
        return 1;

    ShakeUiHostParams shake_ui{};
    shake_ui.accel_step         = shake.accel_step;
    shake_ui.torque_step        = shake.torque_step;
    shake_ui.accel_limit        = shake.accel_limit;
    shake_ui.torque_limit       = shake.torque_limit;
    shake_ui.decay_rate_per_sec = shake.decay_rate_per_sec;
    if(!sim_gpu_upload_shake_ui(shake_ui))
        return 1;

    std::vector<float>         script_times;
    std::vector<unsigned char> script_keys;
    script_times.reserve(scripted_keys.events.size());
    script_keys.reserve(scripted_keys.events.size());
    for(const auto &ev : scripted_keys.events)
    {
        script_times.push_back(ev.first);
        script_keys.push_back(ev.second);
    }
    if(!sim_gpu_upload_scripted_keys(s,
                                     static_cast<int>(script_times.size()),
                                     script_times.data(),
                                     script_keys.data()))
        return 1;

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
        if(!no_output)
            std::printf("[controls] keyboard shaking disabled (stdin is not "
                        "interactive)\n");
    }

    const auto benchmark_init_done = std::chrono::steady_clock::now();

    if(out)
        write_frame(out, s, die, 0, 0.0f);

    const int step_budget =
        (max_steps > 0)
            ? max_steps
            : static_cast<int>(std::ceil(static_cast<double>(TOTAL_TIME) /
                                         static_cast<double>(DT_MIN))) +
                  128;
    constexpr double LIVE_STATS_INTERVAL_S = 0.5;
    const Clock::time_point compute_t0 = Clock::now();
    Clock::time_point       live_log_t0 = compute_t0;
    int                     live_log_step0 = 0;
    int                     live_log_frames = 0;
    int                     live_last_step = 0;
    float                   live_log_sim_t0 = 0.0f;
    float                   live_last_sim_t = 0.0f;
    if(out && !no_output)
    {
        std::printf(
            "[live] fps counter enabled: interval=%.1fs output_every=%d "
            "realtime_output=%d\n",
            LIVE_STATS_INTERVAL_S,
            output_every,
            realtime_output ? 1 : 0);
        std::fflush(stdout);
    }

    float prev_kb[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for(int si = 0; si < step_budget; ++si)
    {
        keyboard.poll();
        SimGpuInputDelta input_delta{};
        input_delta.delta[0] = shake.accel.x - prev_kb[0];
        input_delta.delta[1] = shake.accel.y - prev_kb[1];
        input_delta.delta[2] = shake.accel.z - prev_kb[2];
        input_delta.delta[3] = shake.torque.x - prev_kb[3];
        input_delta.delta[4] = shake.torque.y - prev_kb[4];
        input_delta.delta[5] = shake.torque.z - prev_kb[5];
        input_delta.flags =
            shake.pending_clear_control ? SIM_GPU_SHAKE_CLEAR_CONTROL : 0u;

        bool has_input_delta = input_delta.flags != 0u;
        for(int i = 0; i < 6; ++i)
        {
            if(input_delta.delta[i] != 0.0f)
            {
                has_input_delta = true;
                break;
            }
        }

        prev_kb[0] = shake.accel.x;
        prev_kb[1] = shake.accel.y;
        prev_kb[2] = shake.accel.z;
        prev_kb[3] = shake.torque.x;
        prev_kb[4] = shake.torque.y;
        prev_kb[5] = shake.torque.z;

        shake.pending_clear_control = false;

        step_sim(s, die, has_input_delta ? &input_delta : nullptr);

        if(parity_debug.enabled && parity_debug.stream)
        {
            int   wstep = 0;
            float wt    = 0.0f;
            if(!sim_gpu_pull_frame_for_write(s, die, &wstep, &wt))
                std::abort();
            SimGpuParityDebug dbg{};
            if(!sim_gpu_pull_parity_debug(s, &dbg))
                std::abort();
            const GhostParityStats ghost =
                compute_ghost_parity_stats(s, parity_debug.ghost_eps);
            const std::uint64_t fluid_hash = compute_fluid_state_hash(s);
            emit_parity_log_line(
                parity_debug.stream, wstep, wt, ghost, dbg, fluid_hash);
        }

        if(verbose && !no_output && (si + 1) % LOG_EVERY == 0)
        {
            std::printf("[step %5d] keyboard control=(%+.2f,%+.2f,%+.2f) "
                        "torque=(%+.3f,%+.3f,%+.3f)",
                        si + 1,
                        shake.accel.x,
                        shake.accel.y,
                        shake.accel.z,
                        shake.torque.x,
                        shake.torque.y,
                        shake.torque.z);
            if(keyboard_active)
            {
                std::printf("  steps(accel=%.2f torque=%.3f)",
                            shake.accel_step,
                            shake.torque_step);
            }
            std::printf("\n");
            std::fflush(stdout);
        }

        if(out && ((si + 1) % output_every == 0))
        {
            int   wstep = 0;
            float wt    = 0.0f;
            if(!sim_gpu_pull_frame_for_write(s, die, &wstep, &wt))
                std::abort();
            if(realtime_output)
            {
                const auto target =
                    compute_t0 +
                    std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(wt));
                std::this_thread::sleep_until(target);
            }
            write_frame(out, s, die, wstep, wt);
            ++live_log_frames;
            live_last_step = wstep;
            live_last_sim_t = wt;
        }

        if(out && !no_output)
        {
            const Clock::time_point live_now = Clock::now();
            const double wall_s =
                std::chrono::duration<double>(live_now - live_log_t0).count();
            if(wall_s >= LIVE_STATS_INTERVAL_S)
            {
                const double step_fps =
                    static_cast<double>((si + 1) - live_log_step0) / wall_s;
                const double output_fps =
                    static_cast<double>(live_log_frames) / wall_s;
                const double sim_rate =
                    static_cast<double>(live_last_sim_t - live_log_sim_t0) /
                    wall_s;
                std::printf(
                    "[live] step=%d sim_t=%.3fs sim_rate=%.2fx "
                    "step_fps=%.1f output_fps=%.1f\n",
                    live_last_step,
                    live_last_sim_t,
                    sim_rate,
                    step_fps,
                    output_fps);
                std::fflush(stdout);

                live_log_t0     = live_now;
                live_log_step0  = si + 1;
                live_log_frames = 0;
                live_log_sim_t0 = live_last_sim_t;
            }
        }
    }

    if(out)
        std::fclose(out);
    if(parity_debug.stream)
        std::fclose(parity_debug.stream);
    sim_gpu_shutdown(s);
    const auto benchmark_done = std::chrono::steady_clock::now();
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
    if(benchmark_timings)
    {
        const double init_s =
            std::chrono::duration<double>(benchmark_init_done - benchmark_start)
                .count();
        const double total_s =
            std::chrono::duration<double>(benchmark_done - benchmark_start).count();
        std::printf("benchmark_init_s=%.9f\n", init_s);
        std::printf("benchmark_total_s=%.9f\n", total_s);
        std::fflush(stdout);
    }
    if(verbose && !no_output)
    {
        std::printf("[done] ran %d steps (step cap for TOTAL_TIME ~= %.3f s)\n",
                    step_budget,
                    TOTAL_TIME);
        std::printf("[dump] frames written to '%s'\n", out_path);
    }
    return 0;
}
