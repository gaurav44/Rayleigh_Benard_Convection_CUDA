# Definition of the minimum required cmake Version
cmake_minimum_required(VERSION 3.0)
# Definition of the Project
# Later you can access the project variable like ${CFDLAB_SOURCE_DIR}
project(CFDLAB VERSION 1.0 LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)
add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
)

# Definition of the C++ Standard 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -fstrict-aliasing")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing")

set(CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures to compile for")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -maxrregcount=64 -Xptxas -dlcm=ca")

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

add_executable(main ${SOURCES_CPP} ${SOURCES_CU})

set_target_properties(main PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON
                        CUDA_ARCHITECTURES "61"
)

target_include_directories(main PUBLIC include include/cuda_kernels ${CUDA_TOOLKIT_ROOT_DIR}/include)
