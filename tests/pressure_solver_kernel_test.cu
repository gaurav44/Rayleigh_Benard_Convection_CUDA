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
#include "pressure_solver_kernels.hpp"
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

// Test case for PressureSolverKernelShared
TEST(PressureSolverKernelTest, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 36, jmax = 36;
    int size = 36*36;
    double dx = 0.1, dy = 0.1, gamma=0.5, omg = 1.7;
    double coeff = omg / (2.0 * (1.0 / (dx * dx) +
                               1.0 / (dy * dy)));

    // Set up test arrays for U and V
    double h_P[size], h_RS[size], h_P_expected[size];
    initArray(h_P, size);
    initArray(h_RS, size);
    initArray(h_P_expected, h_P, size);

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
    double *d_P, *d_RS;
    
    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_P, size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_RS, size * sizeof(double)));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_P, h_P, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_RS, h_RS, size * sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE_SOR, BLOCK_SIZE_SOR);
    dim3 numBlocks((imax + BLOCK_SIZE_SOR - 1) / BLOCK_SIZE_SOR,
                   (jmax + BLOCK_SIZE_SOR - 1) / BLOCK_SIZE_SOR);
    PressureSolverKernels::SORKernelShared<<<numBlocks, threadsPerBlock>>>(d_P, d_RS, imax, jmax, omg, coeff, 0);
    CHECK(cudaGetLastError());
    PressureSolverKernels::SORKernelShared<<<numBlocks, threadsPerBlock>>>(d_P, d_RS, imax, jmax, omg, coeff, 1);
    CHECK(cudaGetLastError());
    
    // Copy result back to host
    CHECK(cudaMemcpy(&h_P, d_P, size*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());

    // Manually compute the expected result on host (for verification)
    for (int i = 1; i < imax - 1; i++) {
        for (int j = 1; j < jmax - 1; j++) {
            int idx = j * imax + i;
            if((i + j) % 2 == 0) {
                h_P_expected[idx] =
                (1.0 - omg) * h_P_expected[idx] +
                coeff * (DiscretizationHost::sor_helper(h_P_expected, i, j) - h_RS[idx]);
            }
        }
    }

    for (int i = 1; i < imax - 1; i++) {
        for (int j = 1; j < jmax - 1; j++) {
            int idx = j * imax + i;
            if((i + j) % 2 == 1) {
                h_P_expected[idx] =
                (1.0 - omg) * h_P_expected[idx] +
                coeff * (DiscretizationHost::sor_helper(h_P_expected, i, j) - h_RS[idx]);
            }
        }
    }


    for (int i = 1; i < imax; i++) {
        for(int j = 1; j < jmax; j++) {
            int idx = j * imax + i;
            EXPECT_NEAR(h_P_expected[idx], h_P[idx], 1e-8);
        }
    }
}

// Test case for ResidualKernelShared
TEST(ResidualKernelTest, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 36, jmax = 36;
    int size = 36*36;
    double dx = 0.1, dy = 0.1, gamma=0.5, omg = 1.7;
    double coeff = omg / (2.0 * (1.0 / (dx * dx) +
                               1.0 / (dy * dy)));

    // Set up test arrays for U and V
    double h_P[size], h_RS[size];
    initArray(h_P, size);
    initArray(h_RS, size);

    Discretization d_disc(imax, jmax, dx, dy,
                        gamma);
    
    DiscretizationHost h_disc(imax, jmax, dx, dy,
                        gamma);

    // Device pointers
    double *d_P, *d_RS;
    
    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_P, size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_RS, size * sizeof(double)));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_P, h_P, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_RS, h_RS, size * sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE_RES, BLOCK_SIZE_RES);
    dim3 numBlocks((imax + BLOCK_SIZE_RES - 1) / BLOCK_SIZE_RES,
                   (jmax + BLOCK_SIZE_RES - 1) / BLOCK_SIZE_RES);
    std::vector<double> h_rlocBlock;
    double *d_rlocBlock;
    double h_rloc = 0.0;
    CHECK(cudaMalloc(&d_rlocBlock, numBlocks.x * numBlocks.y * sizeof(double)));
    h_rlocBlock.resize(numBlocks.x * numBlocks.y);

    PressureSolverKernels::residualKernelShared<<<numBlocks, threadsPerBlock>>>(d_P, d_RS, imax, jmax, d_rlocBlock);
    CHECK(cudaGetLastError());

    // Copy result back to host
    CHECK(cudaMemcpy(h_rlocBlock.data(), d_rlocBlock, numBlocks.x*numBlocks.y*sizeof(double), cudaMemcpyDeviceToHost));

    // Find the maximum in the result array
    for (int i = 0; i < numBlocks.x * numBlocks.y; ++i) {
        h_rloc = h_rloc + h_rlocBlock[i];
    }

    // Manually compute the expected result on host (for verification)
    double rloc = 0.0;

    // Using squared value of difference to calculate residual
    for (int i = 1; i < imax - 1; i++) {
        for (int j = 1; j < jmax - 1; j++) {
            int idx = j * imax + i;
            double val = DiscretizationHost::laplacian(h_P, i, j) - h_RS[idx];
            rloc += (val * val);
        }
    }
    h_rloc /= (imax*jmax);
    rloc /= (imax*jmax);
    h_rloc = sqrt(h_rloc);
    rloc = sqrt(rloc);
    EXPECT_NEAR(h_rloc, rloc, 1e-8);
}

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}