#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include "block_sizes.hpp"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "datastructure.hpp"
#include "velocity_kernels.hpp"
#include "cuda_utils.hpp"
#include <iostream>

// Helper function to allocate and initialize data
void initArray(thrust::host_vector<double>& arr, int size) {
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

// Test case for timeStepKernels
TEST(VelocityKernelsTest, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 85, jmax = 18;
    int size = (imax + 2) * (jmax + 2);
    double dx = 0.1, dy = 0.1, dt = 0.01;

    // Host arrays
    thrust::host_vector<double> h_U(size, 0.0), h_V(size, 0.0), h_F(size), h_G(size), h_P(size),
                                h_U_expected(size, 0.0), h_V_expected(size, 0.0);

    // Initialize input data (simple values for easy verification)
    initArray(h_F, size);  
    initArray(h_G, size);  
    initArray(h_P, size);

    thrust::device_vector<double> d_U(size,0.0), d_V(size,0.0), d_F(size), d_G(size), d_P(size);
    thrust::copy(h_F.begin(), h_F.end(), d_F.begin());
    thrust::copy(h_G.begin(), h_G.end(), d_G.begin());
    thrust::copy(h_P.begin(), h_P.end(), d_P.begin());
    CHECK(cudaGetLastError());

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE_UV, BLOCK_SIZE_UV);
    dim3 numBlocks((imax + 2 + BLOCK_SIZE_UV - 1) / BLOCK_SIZE_UV,
                   (jmax + 2 + BLOCK_SIZE_UV - 1) / BLOCK_SIZE_UV);
    
    VelocityKernels::velocityKernelShared<<<numBlocks, threadsPerBlock>>>
    (thrust::raw_pointer_cast(d_U.data()),
     thrust::raw_pointer_cast(d_V.data()),
     thrust::raw_pointer_cast(d_F.data()),
     thrust::raw_pointer_cast(d_G.data()),
     thrust::raw_pointer_cast(d_P.data()), dx, dy, imax+2, jmax+2, dt);
    CHECK(cudaGetLastError());
    
    // Copy result back to host
    thrust::copy(d_U.begin(), d_U.end(), h_U.begin());
    thrust::copy(d_V.begin(), d_V.end(), h_V.begin());
    CHECK(cudaGetLastError());

    for (int i = 1; i < imax; i++) {
        for(int j = 1; j < jmax+1; j++) {
            int idx = j * (imax + 2) + i;
            h_U_expected[idx] = h_F[idx] - dt * (h_P[idx + 1] - h_P[idx]) / dx;
            if(h_U[idx] -  h_U_expected[idx] > 1e-8)
                std::cout << i << " " << j << " " << h_U[idx] << " " << h_U_expected[idx] << "\n";
        }
    }

    for (int i = 1; i < imax+1; i++) {
        for(int j = 1; j < jmax; j++) {
            int idx = j * (imax+2) + i;
            h_V_expected[idx] = h_G[idx] - dt * (h_P[idx + imax + 2] - h_P[idx]) / dy;
            if(h_V[idx] -  h_V_expected[idx] > 1e-8)
                std::cout << i << " " << j << " " << h_V[idx] << " " << h_V_expected[idx] << "\n";
        }
    }

    // Verify that the device result matches the expected result
    for (int j = 1; j < jmax - 1; ++j) {
        for (int i = 1; i < imax - 1; ++i) {
            int idx = j * (imax+2) + i;
            EXPECT_NEAR(h_U_expected[idx], h_U[idx], 1e-8);
            EXPECT_NEAR(h_V_expected[idx], h_V[idx], 1e-8);
        }
    }
}

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}