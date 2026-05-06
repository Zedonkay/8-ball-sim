cmake_minimum_required(VERSION 3.16)
project(sph_simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_GPU_SIM "Build CUDA GPU simulator target" ON)
option(CUDA_ALLOW_UNSUPPORTED_COMPILER "Allow nvcc to use an unsupported host compiler" ON)

set(SPH_SIM_SOURCES
    src/simulator/main.cpp
    src/common/input_controls.cpp
    src/simulator/spatial_hash.cpp
    src/simulator/initialization.cpp
    src/simulator/sim.cpp
)

set(SPH_GPU_SIM_SOURCES
    src/gpu_sim/main.cpp
    src/common/input_controls.cpp
    src/gpu_sim/spatial_hash.cpp
    src/gpu_sim/initialization.cpp
    src/gpu_sim/sim.cpp
)

add_executable(sph_simulator
    ${SPH_SIM_SOURCES}
)

target_include_directories(sph_simulator PRIVATE src/simulator src/common)

add_library(sph_simulator_core STATIC
    src/simulator/initialization.cpp
)

target_include_directories(sph_simulator_core PRIVATE
    src/simulator
    src/common
)

target_compile_options(sph_simulator PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2 -Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

target_compile_options(sph_simulator_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2 -Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# Driver (nvidia-smi) != toolkit: CMake needs nvcc. Search common locations when
# it is not on PATH (e.g. CUDA_HOME set but bin/ not prepended).
if(CMAKE_CUDA_COMPILER AND NOT EXISTS "${CMAKE_CUDA_COMPILER}")
    unset(CMAKE_CUDA_COMPILER CACHE)
endif()
if(NOT CMAKE_CUDA_COMPILER)
    find_program(
        SPH_NVCC_EXECUTABLE
        NAMES nvcc
        HINTS
            ENV CUDA_HOME
            ENV CUDA_PATH
            ENV CUDAToolkit_ROOT
            /usr/local/cuda
        PATH_SUFFIXES bin
        DOC "CUDA toolkit compiler (nvcc)"
    )
    if(SPH_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER "${SPH_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
endif()

if(BUILD_GPU_SIM)
    # CUDA 11.x + GCC 13+ + recent glibc: cicc fails on _Float32/_Float64 in system
    # headers. Use an older g++ for nvcc host code only; .cpp files still use CMAKE_CXX_COMPILER.
    if(UNIX AND NOT APPLE)
        if(NOT CMAKE_CUDA_HOST_COMPILER)
            find_program(
                SPH_CUDA_HOST_CXX_CANDIDATE
                NAMES g++-11 g++-10 g++-12
                DOC "Host compiler for nvcc (.cu host side)"
            )
            if(SPH_CUDA_HOST_CXX_CANDIDATE)
                set(CMAKE_CUDA_HOST_COMPILER "${SPH_CUDA_HOST_CXX_CANDIDATE}"
                    CACHE FILEPATH "Host C++ compiler used by nvcc")
                message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER} (avoids CUDA 11 + new GCC/glibc header issues)")
            else()
                message(WARNING "No g++-11/g++-10/g++-12 found. CUDA 11 with system GCC 13+ often fails (e.g. _Float32 errors). Install g++-11 or set CMAKE_CUDA_HOST_COMPILER.")
            endif()
        else()
            message(STATUS "CUDA host compiler preset: ${CMAKE_CUDA_HOST_COMPILER}")
        endif()
    endif()

    if(CUDA_ALLOW_UNSUPPORTED_COMPILER)
        # Must be set before check_language/CompilerId so detection passes on GCC>11.
        string(APPEND CMAKE_CUDA_FLAGS " --allow-unsupported-compiler")
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        add_executable(sph_simulator_gpu
            ${SPH_GPU_SIM_SOURCES}
            src/gpu_sim/sim_cuda.cu
            src/gpu_sim/sim_cuda_runtime.cu
            src/gpu_sim/sim_cuda_kernels.cu
        )
        target_include_directories(sph_simulator_gpu PRIVATE src/gpu_sim src/common)
        target_compile_options(sph_simulator_gpu PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O2 -Wall -Wextra -Wpedantic>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4 /O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-O2>
        )
        message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER} (target sph_simulator_gpu)")
    else()
        add_custom_target(sph_simulator_gpu
            COMMAND ${CMAKE_COMMAND} -E echo "CUDA compiler not found. Install CUDA toolkit or set CMAKE_CUDA_COMPILER."
        )
        message(STATUS "CUDA compiler not found: skipping sph_simulator_gpu (CPU-only build)")
    endif()
else()
    message(STATUS "BUILD_GPU_SIM=OFF: skipping sph_simulator_gpu target")
endif()

find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE PROJECT_SOURCE_FILES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.cxx"
        "${CMAKE_SOURCE_DIR}/src/*.cc"
        "${CMAKE_SOURCE_DIR}/src/*.c"
        "${CMAKE_SOURCE_DIR}/src/*.cu"
        "${CMAKE_SOURCE_DIR}/src/*.cuh"
        "${CMAKE_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_SOURCE_DIR}/src/*.h"
    )

    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i ${PROJECT_SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-format on project sources"
        VERBATIM
    )

    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${PROJECT_SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Checking clang-format on project sources"
        VERBATIM
    )
else()
    message(STATUS "clang-format not found: 'format' and 'format-check' targets will be unavailable")
endif()
