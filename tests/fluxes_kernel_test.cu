#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include "block_sizes.hpp"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "discretization_host.hpp"
#include "discretization.hpp"
#include "fluxes_kernels.hpp"
#include "cuda_utils.hpp"
#include <iostream>

// Helper function to allocate and initialize data
void initArray(double* arr, int size) {
    // Seed with a real random value, if available
    std::random_device rd;
    // Initialize the random number generator (Mersenne Twister engine)
    std::mt19937 gen(rd());
    // Create a uniform distribution between 1 and 5
    std::uniform_real_distribution<> dis(-5.0, 5.0);
    for (int i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }
}

void initArray(double* arr, double* arr2, int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = arr2[i];
    }
}

// Test case for FluxesKernelShared
TEST(FluxesKernelsTest, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 36, jmax = 36;
    int size = 36*36;
    double dx = 0.1, dy = 0.1, gamma=0.5, alpha = 0.2, dt = 0.05, nu = 0.0296, GX = 1.0, GY = -9.81, beta = 0.00179 ;

    // Set up test arrays for U and V
    double h_U[size], h_V[size], h_T[size], h_F[size], h_G[size], h_F_expected[size], h_G_expected[size];
    initArray(h_U, size);
    initArray(h_V, size);
    initArray(h_T, size);
    initArray(h_F, size);
    initArray(h_F_expected, h_F, size);
    initArray(h_G, size);
    initArray(h_G_expected, h_G, size);

    Discretization d_disc(imax, jmax, dx, dy,
                        gamma);
    
    DiscretizationHost h_disc(imax, jmax, dx, dy,
                        gamma);

    // Device pointers
    double *d_U, *d_V, *d_T, *d_F, *d_G;
    
    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_U, size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_V, size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_T, size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_F, size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_G, size * sizeof(double)));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_U, h_U, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, h_V, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_T, h_T, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_F, h_F, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_G, h_G, size * sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE_FG, BLOCK_SIZE_FG);
    dim3 numBlocks((imax + BLOCK_SIZE_FG - 1) / BLOCK_SIZE_FG,
                   (jmax + BLOCK_SIZE_FG - 1) / BLOCK_SIZE_FG);
    FluxesKernels::FluxesKernelShared<<<numBlocks, threadsPerBlock>>>(d_U, d_V, d_T, d_F, d_G, imax, jmax, nu, dt, GX, GY, beta);
    CHECK(cudaGetLastError());
    
    // Copy result back to host
    CHECK(cudaMemcpy(&h_F, d_F, size*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_G, d_G, size*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());

    // Manually compute the expected result on host (for verification)
    for(int i = 1; i < imax - 2; i++){
        for(int j = 1; j < jmax - 1; j++) {
            int idx = j * imax + i;
            h_F_expected[idx] = h_U[idx] + dt*(nu*DiscretizationHost::diffusion(h_U, i, j) 
                                             - DiscretizationHost::convection_u(h_U, h_V, i, j)) - (beta*dt/2
                                             *(h_T[idx] + h_T[idx+1]))*GX;
        }       
    }

    for (int i = 1; i < imax - 1; i++) {
        for(int j = 1; j < jmax - 2; j++) {
            int idx = j * imax + i;
            h_G_expected[idx] = h_V[idx] + dt*(nu*DiscretizationHost::diffusion(h_V, i, j) 
                                - DiscretizationHost::convection_v(h_U, h_V, i, j)) - (beta*dt/2 
                                *(h_T[idx] + h_T[idx+imax]))*GY;
        }    
    } 

    for (int i = 1; i < imax; i++) {
        for(int j = 1; j < jmax; j++) {
            int idx = j * imax + i;
            // std::cout << h_F_expected[idx] << " " <<"\n";
            EXPECT_NEAR(h_F_expected[idx], h_F[idx], 1e-8);
            EXPECT_NEAR(h_G_expected[idx], h_G[idx], 1e-8);
        }
    }
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_T);
    cudaFree(d_F);
    cudaFree(d_G);
}

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}