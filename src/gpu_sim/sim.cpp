#include "sim.h"

#include "sim_cuda.h"

#include <cstdlib>

void step_sim(SimState    &s,
              RigidDie    &die,
              const SimGpuInputDelta *input_delta)
{
    (void)die;
    if(!sim_gpu_ensure_initialized(s))
        std::abort();
    if(!sim_gpu_advance(s, input_delta))
        std::abort();
}
