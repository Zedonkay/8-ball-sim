#pragma once

#include "rigid_body.h"
#include "sim_state.h"

void poisson_disk_sample_sphere(SimState &s);
void inject_ghost_particles(SimState &s, const RigidDie &die);
