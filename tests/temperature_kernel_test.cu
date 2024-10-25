#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include "block_sizes.hpp"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "temperature_kernels.hpp"
#include "discretization_host.hpp"
#include "discretization.hpp"
#include "cuda_utils.hpp"
#include <iostream>

// Helper function to allocate and initialize data
void initArray(double* arr, int size) {
    // Seed with a real random value, if available
    std::random_device rd;
    // Initialize the random number generator (Mersenne Twister engine)
    std::mt19937 gen(rd());
    // Create a uniform distribution between 1 and 5
    std::uniform_real_distribution<> dis(1.0, 5.0);
    for (int i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }
}

void initArray(double* arr, double* arr2, int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = arr2[i];
    }
}

// Test case for TempratureKernelShared
TEST(TemperatureKernelsTest, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 36, jmax = 36;
    int size = 36*36;
    double dx = 0.1, dy = 0.1, gamma=0.5, alpha = 0.2, dt = 0.05;

    // Set up test arrays for U and V
    double h_U[size], h_V[size], h_T[size], h_Told[size], h_T_expected[size];
    initArray(h_U, size);
    initArray(h_V, size);
    initArray(h_T, size);
    initArray(h_Told, h_T, size);
    initArray(h_T_expected, h_T, size);
    // double h_U[36] = {1.0, 1.2, 1.4, 1.6, 1.8, 1.9,
    //                   2.0, 2.2, 2.4, 2.6, 2.8, 1.6,
    //                   3.0, 3.2, 3.4, 3.6, 3.8, 1.5,
    //                   4.0, 4.2, 4.4, 4.6, 4.8, 1.8,
    //                   5.0, 5.2, 5.4, 5.6, 5.8, 6.0,
    //                   5.0, 5.2, 5.4, 5.6, 5.8, 6.0};
    
    // double h_V[36] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.9,
    //                   1.5, 1.6, 1.7, 1.8, 1.9, 1.6,
    //                   2.0, 2.1, 2.2, 2.3, 2.4, 1.5,
    //                   2.5, 2.6, 2.7, 2.8, 2.9, 1.8,
    //                   3.0, 3.1, 3.2, 3.3, 3.4, 3.6,
    //                   3.0, 3.1, 3.2, 3.3, 3.4, 3.6};

    // double h_T[36] = {1.0, 1.8, 1.2, 1.3, 1.4, 1.9,
    //                   1.5, 1.4, 1.8, 1.8, 1.9, 1.6,
    //                   2.2, 2.7, 2.6, 2.1, 2.0, 1.5,
    //                   2.5, 2.3, 2.5, 2.8, 2.9, 1.8,
    //                   3.0, 3.1, 3.2, 3.3, 3.4, 3.6,
    //                   3.0, 3.1, 3.2, 3.3, 3.4, 3.6};
    
    // double h_Told[36] = {1.0, 1.8, 1.2, 1.3, 1.4, 1.9,
    //                   1.5, 1.4, 1.8, 1.8, 1.9, 1.6,
    //                   2.2, 2.7, 2.6, 2.1, 2.0, 1.5,
    //                   2.5, 2.3, 2.5, 2.8, 2.9, 1.8,
    //                   3.0, 3.1, 3.2, 3.3, 3.4, 3.6,
    //                   3.0, 3.1, 3.2, 3.3, 3.4, 3.6};

    // double h_T_expected[36] = {1.0, 1.8, 1.2, 1.3, 1.4, 1.9,
    //                   1.5, 1.4, 1.8, 1.8, 1.9, 1.6,
    //                   2.2, 2.7, 2.6, 2.1, 2.0, 1.5,
    //                   2.5, 2.3, 2.5, 2.8, 2.9, 1.8,
    //                   3.0, 3.1, 3.2, 3.3, 3.4, 3.6,
    //                   3.0, 3.1, 3.2, 3.3, 3.4, 3.6};

    Discretization d_disc(imax, jmax, dx, dy,
                        gamma);
    
    DiscretizationHost h_disc(imax, jmax, dx, dy,
                        gamma);

    // Device pointers
    double *d_U, *d_V, *d_T, *d_Told;
    
    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_U, size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_V, size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_T, size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_Told, size * sizeof(double)));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_U, h_U, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, h_V, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_T, h_T, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Told, h_Told, size * sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE_TEMP, BLOCK_SIZE_TEMP);
    dim3 numBlocks((imax + BLOCK_SIZE_TEMP - 1) / BLOCK_SIZE_TEMP,
                   (jmax + BLOCK_SIZE_TEMP - 1) / BLOCK_SIZE_TEMP);
    TemperatureKernels::temperatureKernelShared<<<numBlocks, threadsPerBlock>>>(d_U, d_V, d_T, imax, jmax, alpha, dt);
    // TemperatureKernels::temperature_kernel_call<<<numBlocks, threadsPerBlock>>>(d_U, d_V, d_T, d_Told, dx, dy, imax, jmax, gamma, alpha, dt);
    CHECK(cudaGetLastError());
    
    // Copy result back to host
    CHECK(cudaMemcpy(&h_T, d_T, size*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());

    // Manually compute the expected result on host (for verification)
    for (int i = 1; i < imax-1; i++) {
        for(int j = 1; j < jmax-1; j++) {
            int idx = j * imax + i;
            h_T_expected[idx] =
                h_Told[idx] + dt * (alpha * DiscretizationHost::diffusion(h_Told, i, j)
                                    - DiscretizationHost::convection_T(h_U, h_V, h_Told, i, j));
            // if(h_T_expected[idx] - h_T[idx] > 1e-8) 
            //     std::cout << i << " " << j << " " << h_T_expected[idx] << " " << h_T[idx] << "\n";
        }
    }
    CHECK(cudaGetLastError());
    // Verify that the device result matches the expected result
    // for (int j = 0; j < jmax; j++) {
    //     for(int i = 0; i < imax; i++) {
    //         int idx = j * imax + i;
    //         // if(h_T_expected[idx] - h_T[idx] > 1e-8) 
    //         std::cout << h_T_expected[idx] << " ";
    //         // std::cout << i << " " << j << " " << h_T_expected[idx] << " " << h_T[idx] << "\n";
    //         // EXPECT_NEAR(h_T_expected[idx], h_T[idx], 1e-8);
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "**************************************************************************\n";
    for (int j = 0; j < jmax; j++) {
        for(int i = 0; i < imax; i++) {
            int idx = j * imax + i;
            // if(h_T_expected[idx] - h_T[idx] > 1e-8) 
            // std::cout << h_T[idx] << " ";
            // std::cout << i << " " << j << " " << h_T_expected[idx] << " " << h_T[idx] << "\n";
            EXPECT_NEAR(h_T_expected[idx], h_T[idx], 1e-8);
        }
        // std::cout << "\n";
    }
}

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}