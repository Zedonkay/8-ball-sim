#include "input_controls.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "constants.h"

void ShakeInputState::apply_axis_impulse(float &axis, float delta)
{
    const bool opposite_dir = axis * delta < 0.0f;
    if(reduce_opposite_axis_cancellation && opposite_dir)
    {
        axis += opposite_axis_scale * delta;
        return;
    }
    axis += delta;
}

void ShakeInputState::clamp_accel()
{
    accel.x  = std::fmax(-accel_limit, std::fmin(accel.x, accel_limit));
    accel.y  = std::fmax(-accel_limit, std::fmin(accel.y, accel_limit));
    accel.z  = std::fmax(-accel_limit, std::fmin(accel.z, accel_limit));
    torque.x = std::fmax(-torque_limit, std::fmin(torque.x, torque_limit));
    torque.y = std::fmax(-torque_limit, std::fmin(torque.y, torque_limit));
    torque.z = std::fmax(-torque_limit, std::fmin(torque.z, torque_limit));
}

void ShakeInputState::print_input_state() const
{
    if(!output_enabled || (announce_requires_verbose && !SIM_VERBOSE))
        return;
    std::printf(
        "[input] accel=(%+.2f,%+.2f,%+.2f) torque=(%+.3f,%+.3f,%+.3f) "
        "steps(accel=%.2f torque=%.3f)\n",
        accel.x,
        accel.y,
        accel.z,
        torque.x,
        torque.y,
        torque.z,
        accel_step,
        torque_step);
    std::fflush(stdout);
}

void ShakeInputState::handle_key(unsigned char c, bool announce)
{
    bool changed      = false;
    bool step_changed = false;

    if(c == 'w' || c == 'W')
    {
        apply_axis_impulse(accel.y, accel_step);
        changed = true;
    }
    else if(c == 's' || c == 'S')
    {
        apply_axis_impulse(accel.y, -accel_step);
        changed = true;
    }
    else if(c == 'a' || c == 'A')
    {
        apply_axis_impulse(accel.x, -accel_step);
        changed = true;
    }
    else if(c == 'd' || c == 'D')
    {
        apply_axis_impulse(accel.x, accel_step);
        changed = true;
    }
    else if(c == 'q' || c == 'Q')
    {
        apply_axis_impulse(accel.z, accel_step);
        changed = true;
    }
    else if(c == 'e' || c == 'E')
    {
        apply_axis_impulse(accel.z, -accel_step);
        changed = true;
    }
    else if(c == 'i' || c == 'I')
    {
        apply_axis_impulse(torque.x, torque_step);
        changed = true;
    }
    else if(c == 'k' || c == 'K')
    {
        apply_axis_impulse(torque.x, -torque_step);
        changed = true;
    }
    else if(c == 'j' || c == 'J')
    {
        apply_axis_impulse(torque.y, torque_step);
        changed = true;
    }
    else if(c == 'l' || c == 'L')
    {
        apply_axis_impulse(torque.y, -torque_step);
        changed = true;
    }
    else if(c == 'u' || c == 'U')
    {
        apply_axis_impulse(torque.z, torque_step);
        changed = true;
    }
    else if(c == 'o' || c == 'O')
    {
        apply_axis_impulse(torque.z, -torque_step);
        changed = true;
    }
    else if(c == ' ')
    {
        accel                 = {0.0f, 0.0f, 0.0f};
        torque                = {0.0f, 0.0f, 0.0f};
        pending_clear_control = true;
        changed               = true;
    }
    else if(c == '[')
    {
        accel_step   = std::fmax(min_accel_step, accel_step * 0.8f);
        step_changed = true;
    }
    else if(c == ']')
    {
        accel_step   = std::fmin(max_accel_step, accel_step * 1.25f);
        step_changed = true;
    }
    else if(c == ';')
    {
        torque_step  = std::fmax(min_torque_step, torque_step * 0.8f);
        step_changed = true;
    }
    else if(c == '\'')
    {
        torque_step  = std::fmin(max_torque_step, torque_step * 1.25f);
        step_changed = true;
    }

    if(changed || step_changed)
    {
        clamp_accel();
        if(announce)
            print_input_state();
    }
}

void ShakeInputState::decay_inputs(float dt)
{
    const float safe_dt = std::fmax(0.0f, dt);
    const float decay   = std::exp(-decay_rate_per_sec * safe_dt);
    accel               = decay * accel;
    torque              = decay * torque;
}

bool ScriptedKeyEvents::load(const char *path,
                             bool        verbose,
                             bool        output_enabled,
                             bool        warn_malformed_without_verbose)
{
    std::ifstream in(path);
    if(!in)
    {
        std::printf("[error] could not open keys file '%s'\n", path);
        return false;
    }

    std::string line;
    int         line_no = 0;
    while(std::getline(in, line))
    {
        ++line_no;
        std::size_t start = line.find_first_not_of(" \t\r\n");
        if(start == std::string::npos)
            continue;
        if(line[start] == '#')
            continue;

        std::istringstream iss(line.substr(start));
        double             t_double = 0.0;
        std::string        key_token;
        if(!(iss >> t_double >> key_token) || key_token.empty())
        {
            if(output_enabled && (verbose || warn_malformed_without_verbose))
            {
                std::printf("[warn] keys file %s:%d: skip malformed line\n",
                            path,
                            line_no);
            }
            continue;
        }
        events.push_back({static_cast<float>(t_double),
                          static_cast<unsigned char>(key_token[0])});
    }

    std::sort(events.begin(),
              events.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    if(verbose && output_enabled)
    {
        std::printf("[keys-file] loaded %zu events from '%s'\n",
                    events.size(),
                    path);
    }
    return true;
}

void ScriptedKeyEvents::apply_through_time(float t, ShakeInputState &shake)
{
    while(next < events.size() && events[next].first <= t)
    {
        shake.handle_key(events[next].second, false);
        ++next;
    }
}

bool KeyboardControl::init(ShakeInputState *state, bool verbose_logs)
{
    verbose = verbose_logs;
    shake   = state;
    fd      = STDIN_FILENO;
    if(!isatty(fd))
        return false;
    if(tcgetattr(fd, &old_term) != 0)
        return false;
    old_flags = fcntl(fd, F_GETFL, 0);
    if(old_flags < 0)
        return false;

    termios raw = old_term;
    raw.c_lflag &= static_cast<unsigned long>(~(ICANON | ECHO));
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    if(tcsetattr(fd, TCSANOW, &raw) != 0)
        return false;
    if(fcntl(fd, F_SETFL, old_flags | O_NONBLOCK) != 0)
    {
        tcsetattr(fd, TCSANOW, &old_term);
        return false;
    }
    enabled = true;
    return true;
}

void KeyboardControl::shutdown()
{
    if(!enabled)
        return;
    fcntl(fd, F_SETFL, old_flags);
    tcsetattr(fd, TCSANOW, &old_term);
    enabled = false;
}

KeyboardControl::~KeyboardControl()
{
    shutdown();
}

void KeyboardControl::poll()
{
    if(!enabled || !shake)
        return;

    unsigned char buf[64];
    while(true)
    {
        const ssize_t n = read(fd, buf, sizeof(buf));
        if(n <= 0)
            break;
        for(ssize_t i = 0; i < n; ++i)
        {
            const unsigned char c = buf[i];
            if(c == 27 && i + 2 < n && buf[i + 1] == '[')
            {
                bool                changed = false;
                const unsigned char code    = buf[i + 2];
                if(code == 'A')
                {
                    shake->accel.z += shake->accel_step;
                    changed = true;
                }
                else if(code == 'B')
                {
                    shake->accel.z -= shake->accel_step;
                    changed = true;
                }
                else if(code == 'C')
                {
                    shake->accel.x += shake->accel_step;
                    changed = true;
                }
                else if(code == 'D')
                {
                    shake->accel.x -= shake->accel_step;
                    changed = true;
                }
                if(changed)
                {
                    shake->clamp_accel();
                    shake->print_input_state();
                }
                i += 2;
                continue;
            }
            shake->handle_key(c);
        }
    }

    shake->clamp_accel();
}

void enable_high_agitation_profile(ShakeInputState &shake)
{
    shake.decay_rate_per_sec                = 2.0f;
    shake.reduce_opposite_axis_cancellation = true;
    shake.opposite_axis_scale               = 0.3f;

    RIGID_MAX_FORCE_RUNTIME     = 120.0f;
    RIGID_MAX_TORQUE_RUNTIME    = 2.5f;
    RIGID_MAX_SPEED_RUNTIME     = 1.2f;
    RIGID_MAX_OMEGA_RUNTIME     = 55.0f;
    DIE_LINEAR_DAMPING_RUNTIME  = 1.2f;
    DIE_ANGULAR_DAMPING_RUNTIME = 3.0f;
    DIE_CENTER_SPRING_RUNTIME   = 0.0f;
    DIE_CENTER_DAMPING_RUNTIME  = 0.0f;
    DIE_CONTROL_ACCEL_GAIN_RUNTIME = 0.0f;
    DIE_KEY_VELOCITY_GAIN_RUNTIME = 0.02f;
    DIE_KEY_OMEGA_GAIN_RUNTIME    = 260.0f;
}

void enable_stable_live_profile(ShakeInputState &shake)
{
    shake.accel_step                        = 12.0f;
    shake.accel_limit                       = 45.0f;
    shake.torque_step                       = 0.014f;
    shake.torque_limit                      = 0.22f;
    shake.decay_rate_per_sec                = 5.0f;
    shake.max_accel_step                    = 40.0f;
    shake.max_torque_step                   = 0.25f;
    shake.reduce_opposite_axis_cancellation = true;
    shake.opposite_axis_scale               = 0.2f;
    shake.announce_requires_verbose         = false;

    RIGID_MAX_FORCE_RUNTIME     = 10.0f;
    RIGID_MAX_TORQUE_RUNTIME    = 0.18f;
    RIGID_MAX_SPEED_RUNTIME     = 0.45f;
    RIGID_MAX_OMEGA_RUNTIME     = 18.0f;
    DIE_LINEAR_DAMPING_RUNTIME  = 18.0f;
    DIE_ANGULAR_DAMPING_RUNTIME = 28.0f;
    DIE_CENTER_SPRING_RUNTIME   = 22.0f;
    DIE_CENTER_DAMPING_RUNTIME  = 8.0f;
    DIE_CONTROL_ACCEL_GAIN_RUNTIME = 12.0f;
    DIE_KEY_VELOCITY_GAIN_RUNTIME = 0.04f;
    DIE_KEY_OMEGA_GAIN_RUNTIME    = 420.0f;
}

void print_keyboard_controls()
{
    std::printf("[controls] keyboard input enabled:\n"
                "           die accel:  WASD = x/y, Q/E or Up/Down = z\n"
                "           die torque:  I/K = +x/-x, J/L = +y/-y, U/O = "
                "+z/-z\n"
                "           Space = reset die accel + torque\n"
                "           steps:       [ / ] accel step down/up, ; / ' "
                "torque step down/up\n");
    std::fflush(stdout);
}
