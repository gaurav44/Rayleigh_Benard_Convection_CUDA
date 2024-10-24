#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include "block_sizes.hpp"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "datastructure.hpp"
#include "right_hand_side_kernels.hpp"
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

// Test case for rightHandSideKernelShared
TEST(RightHandSideKernelTest, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 85, jmax = 18;
    int size = imax * jmax;
    double dx = 0.1, dy = 0.1, dt = 0.01;

    // Host arrays
    thrust::host_vector<double> h_F(size), h_G(size), h_RS(size, 0.0), h_RS_expected(size);

    // Initialize input data (simple values for easy verification)
    initArray(h_F, size);  // e.g., constant value 1.0
    initArray(h_G, size);  // e.g., constant value 2.0

    thrust::device_vector<double> d_F(size), d_G(size), d_RS(size);
    thrust::copy(h_F.begin(), h_F.end(), d_F.begin());
    thrust::copy(h_G.begin(), h_G.end(), d_G.begin());
    thrust::copy(h_RS.begin(), h_RS.end(), d_RS.begin());
    CHECK(cudaGetLastError());

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE_RS, BLOCK_SIZE_RS);
    dim3 numBlocks((imax + BLOCK_SIZE_RS - 1) / BLOCK_SIZE_RS,
                   (jmax + BLOCK_SIZE_RS - 1) / BLOCK_SIZE_RS);
    RightHandSideKernels::rightHandSideKernelShared<<<numBlocks, threadsPerBlock>>>
    (thrust::raw_pointer_cast(d_F.data()),
     thrust::raw_pointer_cast(d_G.data()),
     thrust::raw_pointer_cast(d_RS.data()), dx, dy, imax, jmax, dt);
    CHECK(cudaGetLastError());
    
    // Copy result back to host
    thrust::copy(d_RS.begin(), d_RS.end(), h_RS.begin());
    CHECK(cudaGetLastError());

    // Manually compute the expected result on host (for verification)
    for (int j = 1; j < jmax - 1; ++j) {
        for (int i = 1; i < imax - 1; ++i) {
            int idx = j * imax + i;
            double term1 = (h_F[idx] - h_F[idx - 1]) / dx;
            double term2 = (h_G[idx] - h_G[idx - imax]) / dy;
            h_RS_expected[idx] = (term1 + term2) / dt;
        }
    }
    CHECK(cudaGetLastError());
    // Verify that the device result matches the expected result
    for (int i = 0; i < size ; ++i) {
        EXPECT_NEAR(h_RS[i], h_RS_expected[i], 1e-8);
    }
}

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}