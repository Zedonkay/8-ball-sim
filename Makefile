BUILD_DIR_CPU ?= build/cpu
BUILD_DIR_GPU ?= build/gpu
# Prefer an older host compiler for CUDA 11 when present.
CUDA_HOST_COMPILER ?= $(shell command -v g++-11 2>/dev/null)
# CUDA 11 + default GCC 13: use --allow-unsupported-compiler and pin host compiler if found.
# Override if needed, e.g.:
# GPU_CMAKE_ARGS='-DCUDA_ALLOW_UNSUPPORTED_COMPILER=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11'
GPU_CMAKE_ARGS ?= -DCUDA_ALLOW_UNSUPPORTED_COMPILER=ON
ifneq ($(strip $(CUDA_HOST_COMPILER)),)
GPU_CMAKE_ARGS += -DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER)
endif

.PHONY: all GPU configure configure-cpu configure-gpu build build-cpu build-gpu format format-check clean

all: build-cpu

GPU: build-gpu

configure: configure-cpu configure-gpu

configure-cpu:
	cmake -S . -B $(BUILD_DIR_CPU) -DBUILD_GPU_SIM=OFF

configure-gpu:
	cmake -S . -B $(BUILD_DIR_GPU) -DBUILD_GPU_SIM=ON $(GPU_CMAKE_ARGS)

build: build-cpu build-gpu

build-cpu: configure-cpu
	cmake --build $(BUILD_DIR_CPU) --target sph_simulator

build-gpu: configure-gpu
	cmake --build $(BUILD_DIR_GPU) --target sph_simulator_gpu

format: configure
	cmake --build $(BUILD_DIR_CPU) --target format

format-check: configure
	cmake --build $(BUILD_DIR_CPU) --target format-check

clean:
	rm -rf build
