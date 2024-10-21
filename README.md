# Rayleigh-Bénard Convection Simulation with CUDA

This project simulates **Rayleigh-Bénard Convection**, a fundamental fluid dynamics phenomenon, using CUDA to leverage GPU acceleration for high-performance computing. The simulation showcases the formation of convective rolls in a fluid heated from below and cooled from above.

## Hardware requirements
The project requires CUDA-enabled GPU to run. It has been tested with NVIDIA-GeForce GTX 1050 which has Pascal architecture and Compute Capability of 6.1

## Software pre-requisites
- ```CMake``` minimum version ```3.0```
- NVIDIA CUDA Toolkit (version ```11.0``` or later)
- ```C++ 17``` compatible compiler

## Steps to run
- Clone the repository
- ```mkdir build && cd build```
- ```cmake ..``` and then ```make -j``` to build the project
- To run, place the domain.txt in ```build/``` directory and run using ```./main```

