# Definition of the minimum required cmake Version
cmake_minimum_required(VERSION 3.0)
# Definition of the Project
# Later you can access the project variable like ${CFDLAB_SOURCE_DIR}
project(CFDLAB VERSION 1.0 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# Definition of the C++ Standard 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")

set(CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures to compile for")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# file(GLOB files src/*.cpp)
set(SOURCES_CPP
    src/main.cpp
)

set(SOURCES_CU
    src/DataStructure.cu
    src/Simulation.cu
)

add_executable(main ${SOURCES_CPP} ${SOURCES_CU})

set_target_properties(main PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON
                        CUDA_ARCHITECTURES "61"
)

target_include_directories(main PUBLIC include ${CUDA_TOOLKIT_ROOT_DIR}/include)
