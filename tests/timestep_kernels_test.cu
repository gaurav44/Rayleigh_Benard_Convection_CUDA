#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include "block_sizes.hpp"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include "datastructure.hpp"
#include "timestep_kernels.hpp"
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
TEST(TimeStepKernelsTest, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 35, jmax = 35;
    int size = imax * jmax;
    double dx = 0.1, dy = 0.1, dt = 0.01;

    // Set up test arrays for U and V
    thrust::host_vector<double> h_U(size, 0.0);
    thrust::host_vector<double> h_V(size, 0.0);

    h_U[5*35+6] = 5.0;
    h_V[5*35+6] = 7.0;

    // Host arrays
    double h_uMaxExpected, h_vMaxExpected;

    thrust::device_vector<double> d_U(size), d_V(size);
    thrust::copy(h_U.begin(), h_U.end(), d_U.begin());
    thrust::copy(h_V.begin(), h_V.end(), d_V.begin());
    CHECK(cudaGetLastError());

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE_DT, BLOCK_SIZE_DT);
    dim3 numBlocks((imax + BLOCK_SIZE_DT - 1) / BLOCK_SIZE_DT,
                   (jmax + BLOCK_SIZE_DT - 1) / BLOCK_SIZE_DT);

    double *h_uBlockMax;
    double *h_vBlockMax;
    double *d_uBlockMax;
    double *d_vBlockMax;
    double h_uMax=0.0;
    double h_vMax=0.0;

    CHECK(cudaMalloc(&d_uBlockMax, numBlocks.x * numBlocks.y * sizeof(double)));
    CHECK(cudaMalloc(&d_vBlockMax, numBlocks.x * numBlocks.y * sizeof(double)));

    h_uBlockMax = new double[numBlocks.x * numBlocks.y];
    h_vBlockMax = new double[numBlocks.x * numBlocks.y];
    
    TimestepKernels::velocityUMaxKernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_U.data()), imax, jmax, d_uBlockMax);
    CHECK(cudaGetLastError());
    TimestepKernels::velocityVMaxKernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_V.data()), imax, jmax, d_vBlockMax);
    CHECK(cudaMemcpy(h_uBlockMax, d_uBlockMax,
                     numBlocks.x * numBlocks.y * sizeof(double),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_vBlockMax, d_vBlockMax,
                     numBlocks.x * numBlocks.y * sizeof(double),
                     cudaMemcpyDeviceToHost));

    // Find the maximum in the result array
    for (int i = 0; i < numBlocks.x * numBlocks.y; ++i) {
        h_uMax = fmax(h_uMax, h_uBlockMax[i]);
        h_vMax = fmax(h_vMax, h_vBlockMax[i]);
    }
    
    // find expected max values
    auto max_iterU = thrust::max_element(h_U.begin(), h_U.end());
    auto max_iterV = thrust::max_element(h_V.begin(), h_V.end());
    h_uMaxExpected = *max_iterU;
    h_vMaxExpected = *max_iterV;

    // Verify that the device result matches the expected result
    EXPECT_NEAR(h_uMaxExpected, h_uMax, 1e-8);
    EXPECT_NEAR(h_vMaxExpected, h_vMax, 1e-8);
}

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}