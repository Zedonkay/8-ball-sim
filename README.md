# 8-ball-sim

Particle-based (SPH) fluid simulation in a spherical container with a rigid die, with both CPU and CUDA GPU simulation paths plus visualization.

## Run

### 1) Build

CPU:

```bash
make build-cpu
```

GPU (requires CUDA toolkit + compatible NVIDIA setup):

```bash
make build-gpu
```

### 2) Run a simulation

CPU simulation (writes `sim.bin`):

```bash
./build/cpu/sph_simulator sim.bin
```

GPU simulation (writes `sim.bin`):

```bash
./build/gpu/sph_simulator_gpu sim.bin
```

### 3) View output

Native fast viewer (OpenGL/GLUT, reads frames from stdin):

```bash
./build/cpu/fast_viewer < sim.bin
```

## High-level code layout

- `src/simulator/`: CPU SPH solver, rigid die coupling, initialization, CLI runner.
- `src/gpu_sim/`: CUDA-accelerated solver and runtime with a compatible CLI runner.
- `src/common/`: shared keyboard/scripted input control logic.
- `src/vis/`: visualization tools (`visualize.py` and `fast_viewer.cpp`).
- `CMakeLists.txt` + `Makefile`: build configuration and convenience targets.

## Learn more

To learn more, see the full report:
<https://zedonkay.github.io/rigging-magic/final-report.html>