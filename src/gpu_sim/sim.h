#pragma once

#include "rigid_body.h"
#include "sim_state.h"

struct SimGpuInputDelta;

void step_sim(SimState    &s,
              RigidDie    &die,
              const SimGpuInputDelta *input_delta = nullptr);
