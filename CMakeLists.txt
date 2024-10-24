# Definition of the minimum required CMake Version
cmake_minimum_required(VERSION 3.0)

# Definition of the Project
project(RBC_CUDA VERSION 1.0 LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()

# Copy compile_commands.json to the project root for use with editors/tools
add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
)

find_package(CUDA)

# Set the C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -fstrict-aliasing")

set(CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures to compile for")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -maxrregcount=64 -Xptxas -dlcm=ca")

# Source files
set(SOURCES_CPP
    src/main.cpp
    src/simulation.cpp
    src/boundary.cpp
    src/pressure_solver.cpp
)

set(SOURCES_CU
    src/cuda_src/datastructure.cu
    src/cuda_src/temperature_kernels.cu
    src/cuda_src/fluxes_kernels.cu
    src/cuda_src/right_hand_side_kernels.cu
    src/cuda_src/boundary_kernels.cu
    src/cuda_src/discretization.cu
    src/cuda_src/pressure_solver_kernels.cu
    src/cuda_src/velocity_kernels.cu
    src/cuda_src/timestep_kernels.cu
)

# Add the executable for the main application
add_executable(main ${SOURCES_CPP})

# Create a library for the CUDA kernels
add_library(kernels STATIC ${SOURCES_CU})

# Set CUDA-specific properties for the library
set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)

# Include directories for both main and kernels
target_include_directories(kernels PUBLIC include include/cuda_kernels ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(main PUBLIC include include/cuda_kernels ${CUDA_TOOLKIT_ROOT_DIR}/include)

add_subdirectory(tests)

# Link the kernels library to the main executable
target_link_libraries(main kernels)


