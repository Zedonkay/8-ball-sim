#pragma once

#include "vec_math.h"

// Math
constexpr float PI = 3.14159265358979323846f;

// Smoothing and discretization
constexpr float H          = 0.05f;
constexpr float PARTICLE_R = H / 2.0f;
constexpr float PARTICLE_M = 0.008f; // kg
constexpr float SPHERE_R   = 0.15f;  // m, inner radius of spherical container
constexpr float DEFAULT_FLUID_DENSITY = 1000.0f; // kg/m^3
constexpr int   N_PARTICLES           = 3500; // target count for per-step work

// Runtime-tunable fluid coefficients
struct Params
{
    float viscosity        = 0.08f; // mu [Pa*s]
    float rest_density     = DEFAULT_FLUID_DENSITY; // rho_0 [kg/m^3]
    float stiffness        = 12000.0f; // Tait pressure coefficient [Pa]
    float wall_restitution = 0.0f; // sphere bounce back
    float drag_coeff       = 0.08f; // Cd
    float friction         = 0.05f; // beta
};

// Derived/fixed physics
constexpr int   GAMMA                     = 7; // Tait exponent (water-like)
constexpr Vec3  GRAVITY                   = {0.f, 0.f, -9.81f}; // m/s^2
constexpr float CFL_LAMBDA                = 0.7f; // CFL safety factor
constexpr float CFL_SOUND_SPEED_VEL_SCALE = 3.0f;
constexpr float DT_MAX                    = 0.0015f;
constexpr float DT_MIN                    = 2e-5f; 
constexpr float MIN_R2                    = 1e-12f; // avoid division by zero
constexpr float MIN_RHO                   = 1e-6f; // avoid division by zero

// Kernel normalization constants 
constexpr float H6          = H * H * H * H * H * H;
constexpr float H9          = H6 * H * H * H;
constexpr float ALPHA_POLY6 = 315.0f / (64.0f * PI * H9);
constexpr float ALPHA_SPIKY = 45.0f / (PI * H6);
constexpr float ALPHA_VISC  = 45.0f / (PI * H6);

// Spatial hash
constexpr float CELL_SIZE = H;
// Next prime above 2 * N_MAX (16384), reduces hash collisions.
constexpr int TABLE_SIZE = 16411;

// Memory layout
constexpr int N_MAX         = 8192;
constexpr int MAX_NEIGHBORS = 128;
constexpr int PING          = 0;
constexpr int PONG          = 1;

// Coupling/contact constants
constexpr float XSPH_EPS            = 0.75f; // velocity smoothing
constexpr float CONTACT_K           = 800.0f; // penalty spring
constexpr float DIE_CONTACT_MARGIN  = 0.45f * PARTICLE_R; // early contact shell
constexpr float DIE_CONTACT_DAMPING = 12.0f; // N*s/m
constexpr float PARTICLE_AREA       = PI * PARTICLE_R * PARTICLE_R; // m^2
constexpr float DIE_SIDE            = 0.064f; // m
constexpr float DIE_HALF            = DIE_SIDE / 2.0f;
constexpr float DIE_VOLUME          = DIE_SIDE * DIE_SIDE * DIE_SIDE; // m^3
constexpr float DIE_WALL_CLEARANCE  = PARTICLE_R; // keep die from pinching wall particles
constexpr float DIE_GHOST_SURFACE_INSET = 0.1f * PARTICLE_R;
constexpr float DIE_GHOST_MASS_SCALE    = 1.0f;
constexpr float neutral_buoyancy_mass(float fluid_density)
{
    return fluid_density * DIE_VOLUME;
}
constexpr float neutral_buoyancy_mass(const Params &params)
{
    return neutral_buoyancy_mass(params.rest_density);
}
constexpr float NEUTRAL_BUOYANCY_MASS =
    DEFAULT_FLUID_DENSITY * DIE_VOLUME; // rho_fluid * V
constexpr float DIE_MASS       = DEFAULT_FLUID_DENSITY * DIE_VOLUME;
constexpr int   MAX_DIE_GHOSTS = 512;

// Coupling and rigid-body caps.
constexpr float RIGID_MAX_FORCE     = 40.0f;
constexpr float RIGID_MAX_TORQUE    = 0.7f;
constexpr float RIGID_MAX_SPEED     = 0.35f; // m/s
constexpr float RIGID_MAX_OMEGA     = 20.0f; // rad/s
constexpr float DIE_LINEAR_DAMPING  = 4.0f; // 1/s
constexpr float DIE_ANGULAR_DAMPING = 12.0f; // 1/s
constexpr float DIE_CENTER_SPRING   = 0.0f; // N/m, opt-in live stabilizer
constexpr float DIE_CENTER_DAMPING  = 0.0f; // N*s/m, opt-in live stabilizer
constexpr float DIE_CONTROL_ACCEL_GAIN = 0.0f; // opt-in live input boost
constexpr float DIE_KEY_VELOCITY_GAIN = 0.02f; // m/s per accel-step unit
constexpr float DIE_KEY_OMEGA_GAIN    = 260.0f; // rad/s per torque-step unit
constexpr float DIE_IDLE_ACCEL_CONTROL_SLEEP  = 0.35f;
constexpr float DIE_IDLE_TORQUE_CONTROL_SLEEP = 0.003f;
constexpr float DIE_IDLE_GRACE_TIME           = 0.35f;
constexpr float DIE_IDLE_LINEAR_DAMPING       = 45.0f; // extra damping with no input
constexpr float DIE_IDLE_ANGULAR_DAMPING      = 75.0f;
constexpr float DIE_IDLE_FORCE_SLEEP          = 1.5f; // ignore contact jitter at rest
constexpr float DIE_IDLE_TORQUE_SLEEP         = 0.035f;
constexpr float DIE_IDLE_SPEED_SLEEP          = 0.012f;
constexpr float DIE_IDLE_OMEGA_SLEEP          = 0.22f;

// Runtime overrides used by opt-in profiles.
inline float RIGID_MAX_FORCE_RUNTIME     = RIGID_MAX_FORCE;
inline float RIGID_MAX_TORQUE_RUNTIME    = RIGID_MAX_TORQUE;
inline float RIGID_MAX_SPEED_RUNTIME     = RIGID_MAX_SPEED;
inline float RIGID_MAX_OMEGA_RUNTIME     = RIGID_MAX_OMEGA;
inline float DIE_LINEAR_DAMPING_RUNTIME  = DIE_LINEAR_DAMPING;
inline float DIE_ANGULAR_DAMPING_RUNTIME = DIE_ANGULAR_DAMPING;
inline float DIE_CENTER_SPRING_RUNTIME   = DIE_CENTER_SPRING;
inline float DIE_CENTER_DAMPING_RUNTIME  = DIE_CENTER_DAMPING;
inline float DIE_CONTROL_ACCEL_GAIN_RUNTIME = DIE_CONTROL_ACCEL_GAIN;
inline float DIE_KEY_VELOCITY_GAIN_RUNTIME = DIE_KEY_VELOCITY_GAIN;
inline float DIE_KEY_OMEGA_GAIN_RUNTIME    = DIE_KEY_OMEGA_GAIN;

// Visualization (used by src/vis/visualize.py)
constexpr float VIS_FLUID_ALPHA = 0.1f;
constexpr float VIS_GHOST_ALPHA = 0.25f;

// Simulation control
constexpr int   N_REBUILD             = 10; // rebuild neighbors every N steps
constexpr int   WARMUP_STEPS          = 10; // density-only startup passes
constexpr int   LOG_EVERY             = 1; // console logging interval [steps]
constexpr float TOTAL_TIME            = 5.0f; // simulation duration [s]
constexpr int   POISSON_GIVE_UP_AFTER = 1000; // max attempts/sample

// Runtime CLI switches for console noise (--no-output, --verbose).
inline bool SIM_NO_OUTPUT = false;
inline bool SIM_VERBOSE   = false;
