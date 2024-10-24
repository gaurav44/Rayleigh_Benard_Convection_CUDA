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
#include "discretization.hpp"
#include "discretization_host.hpp"
#include "cuda_utils.hpp"
#include <iostream>

namespace DiscretizationTests{
    __global__ void convectionUTest(double* U, double* V, double* result, int i, int j, int imax) {
        *result = Discretization::convection_uSharedMem(U, V, i, j, imax);
    } 
    __global__ void convectionVTest(double* U, double* V, double* result, int i, int j, int imax) {
        *result = Discretization::convection_vSharedMem(U, V, i, j, imax);
    }  
    __global__ void convectiontTest(double* U, double* V, double* T, double* result, int i, int j, int imax) {
        *result = Discretization::convection_TSharedMem(U, V, T, i, j, imax);
    } 
    __global__ void diffusionTest(double* U, double* result, int i, int j, int imax) {
        *result = Discretization::diffusionSharedMem(U, i, j, imax);
    } 
    __global__ void laplacianTest(double* U, double* result, int i, int j, int imax) {
        *result = Discretization::laplacianSharedMem(U, i, j, imax);
    }  
    __global__ void sorHelperTest(double* U, double* result, int i, int j, int imax) {
        *result = Discretization::sor_helperSharedMem(U, i, j, imax);
    } 
    __global__ void interpolateTest(double* U, double* result, int i, int j, int i_offset, int j_offset, int imax) {
        *result = Discretization::interpolateSharedMem(U, i, j, i_offset, j_offset, imax);
    }     
}

// Test case for convection_Kernels
TEST(ConvectionTests, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 5, jmax = 5;
    double dx = 0.1, dy = 0.1, gamma=0.5;

     // Set up test arrays for U and V
    double h_U[25] = {1.0, 1.2, 1.4, 1.6, 1.8,
                      2.0, 2.2, 2.4, 2.6, 2.8,
                      3.0, 3.2, 3.4, 3.6, 3.8,
                      4.0, 4.2, 4.4, 4.6, 4.8,
                      5.0, 5.2, 5.4, 5.6, 5.8};
    
    double h_V[25] = {1.0, 1.1, 1.2, 1.3, 1.4,
                      1.5, 1.6, 1.7, 1.8, 1.9,
                      2.0, 2.1, 2.2, 2.3, 2.4,
                      2.5, 2.6, 2.7, 2.8, 2.9,
                      3.0, 3.1, 3.2, 3.3, 3.4};

    double h_T[25] = {1.0, 1.8, 1.2, 1.3, 1.4,
                      1.5, 1.4, 1.8, 1.8, 1.9,
                      2.2, 2.7, 2.6, 2.1, 2.0,
                      2.5, 2.3, 2.5, 2.8, 2.9,
                      3.0, 3.1, 3.2, 3.3, 3.4};
    
    // Test convection_u for a specific (i, j) point
    int i = 2;  // X-index
    int j = 2;  // Y-index

    Discretization d_disc(imax, jmax, dx, dy,
                        gamma);
    
    DiscretizationHost h_disc(imax, jmax, dx, dy,
                        gamma);

    double uResult = DiscretizationHost::convection_u(h_U, h_V, i, j);
    double vResult = DiscretizationHost::convection_v(h_U, h_V, i, j);
    double tResult = DiscretizationHost::convection_T(h_U, h_V, h_T, i, j);

    // Device pointers
    double *d_U, *d_V, *d_T, *d_uResult, *d_vResult, *d_tResult;
    
    // Allocate device memory
    cudaMalloc((void**)&d_U, 25 * sizeof(double));
    cudaMalloc((void**)&d_V, 25 * sizeof(double));
    cudaMalloc((void**)&d_T, 25 * sizeof(double));
    cudaMalloc((void**)&d_uResult, sizeof(double));
    cudaMalloc((void**)&d_vResult, sizeof(double));
    cudaMalloc((void**)&d_tResult, sizeof(double));
    
    // Copy data from host to device
    cudaMemcpy(d_U, h_U, 25 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, 25 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, h_T, 25 * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel to compute convection_u
    DiscretizationTests::convectionUTest<<<1, 1>>>(d_U, d_V, d_uResult, i, j, imax);
    DiscretizationTests::convectionVTest<<<1, 1>>>(d_U, d_V, d_vResult, i, j, imax);
    DiscretizationTests::convectiontTest<<<1, 1>>>(d_U, d_V, d_T, d_tResult, i, j, imax);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back to host
    double h_uResult, h_vResult, h_tResult;
    cudaMemcpy(&h_uResult, d_uResult, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vResult, d_vResult, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tResult, d_tResult, sizeof(double), cudaMemcpyDeviceToHost);
    
    EXPECT_NEAR(h_uResult, uResult, 1e-8);
    EXPECT_NEAR(h_vResult, vResult, 1e-8);
    EXPECT_NEAR(h_tResult, tResult, 1e-8);
}

// Test case for diffusion_Kernel
TEST(DiffusionTests, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 5, jmax = 5;
    double dx = 0.1, dy = 0.1, gamma=0.5;

     // Set up test arrays for U and V
    double h_U[25] = {1.0, 1.2, 1.4, 1.6, 1.8,
                      2.0, 2.2, 2.4, 2.6, 2.8,
                      3.0, 3.2, 3.4, 3.6, 3.8,
                      4.0, 4.2, 4.4, 4.6, 4.8,
                      5.0, 5.2, 5.4, 5.6, 5.8};
    
    // Test convection_u for a specific (i, j) point
    int i = 2;  // X-index
    int j = 2;  // Y-index

    Discretization d_disc(imax, jmax, dx, dy,
                        gamma);
    
    DiscretizationHost h_disc(imax, jmax, dx, dy,
                        gamma);

    double uResult = DiscretizationHost::diffusion(h_U, i, j);

    // Device pointers
    double *d_U, *d_uResult;
    
    // Allocate device memory
    cudaMalloc((void**)&d_U, 25 * sizeof(double));
    cudaMalloc((void**)&d_uResult, sizeof(double));
    
    // Copy data from host to device
    cudaMemcpy(d_U, h_U, 25 * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel to compute convection_u
    DiscretizationTests::diffusionTest<<<1, 1>>>(d_U, d_uResult, i, j, imax);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back to host
    double h_uResult;
    cudaMemcpy(&h_uResult, d_uResult, sizeof(double), cudaMemcpyDeviceToHost);
    
    EXPECT_NEAR(h_uResult, uResult, 1e-8);
}

// Test case for laplacian_Kernel
TEST(LaplacianTests, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 5, jmax = 5;
    double dx = 0.1, dy = 0.1, gamma=0.5;

     // Set up test arrays for U and V
    double h_U[25] = {1.0, 1.2, 1.4, 1.6, 1.8,
                      2.0, 2.2, 2.4, 2.6, 2.8,
                      3.0, 3.2, 3.4, 3.6, 3.8,
                      4.0, 4.2, 4.4, 4.6, 4.8,
                      5.0, 5.2, 5.4, 5.6, 5.8};
    
    // Test convection_u for a specific (i, j) point
    int i = 2;  // X-index
    int j = 2;  // Y-index

    Discretization d_disc(imax, jmax, dx, dy,
                        gamma);
    
    DiscretizationHost h_disc(imax, jmax, dx, dy,
                        gamma);

    double uResult = DiscretizationHost::laplacian(h_U, i, j);

    // Device pointers
    double *d_U, *d_uResult;
    
    // Allocate device memory
    cudaMalloc((void**)&d_U, 25 * sizeof(double));
    cudaMalloc((void**)&d_uResult, sizeof(double));
    
    // Copy data from host to device
    cudaMemcpy(d_U, h_U, 25 * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel to compute convection_u
    DiscretizationTests::laplacianTest<<<1, 1>>>(d_U, d_uResult, i, j, imax);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back to host
    double h_uResult;
    cudaMemcpy(&h_uResult, d_uResult, sizeof(double), cudaMemcpyDeviceToHost);
    
    EXPECT_NEAR(h_uResult, uResult, 1e-8);
}

// Test case for sor_helper_Kernel
TEST(SORHelperTests, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 5, jmax = 5;
    double dx = 0.1, dy = 0.1, gamma=0.5;

     // Set up test arrays for U and V
    double h_U[25] = {1.0, 1.2, 1.4, 1.6, 1.8,
                      2.0, 2.2, 2.4, 2.6, 2.8,
                      3.0, 3.2, 3.4, 3.6, 3.8,
                      4.0, 4.2, 4.4, 4.6, 4.8,
                      5.0, 5.2, 5.4, 5.6, 5.8};
    
    // Test convection_u for a specific (i, j) point
    int i = 2;  // X-index
    int j = 2;  // Y-index

    Discretization d_disc(imax, jmax, dx, dy,
                        gamma);
    
    DiscretizationHost h_disc(imax, jmax, dx, dy,
                        gamma);

    double uResult = DiscretizationHost::sor_helper(h_U, i, j);

    // Device pointers
    double *d_U, *d_uResult;
    
    // Allocate device memory
    cudaMalloc((void**)&d_U, 25 * sizeof(double));
    cudaMalloc((void**)&d_uResult, sizeof(double));
    
    // Copy data from host to device
    cudaMemcpy(d_U, h_U, 25 * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel to compute convection_u
    DiscretizationTests::sorHelperTest<<<1, 1>>>(d_U, d_uResult, i, j, imax);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back to host
    double h_uResult;
    cudaMemcpy(&h_uResult, d_uResult, sizeof(double), cudaMemcpyDeviceToHost);
    
    EXPECT_NEAR(h_uResult, uResult, 1e-8);
}

// Test case for interpolate_Kernel
TEST(InterpolateTests, HandlesBasicInput) {
    // Define grid size and block size
    int imax = 5, jmax = 5;
    double dx = 0.1, dy = 0.1, gamma=0.5;

     // Set up test arrays for U and V
    double h_U[25] = {1.0, 1.2, 1.4, 1.6, 1.8,
                      2.0, 2.2, 2.4, 2.6, 2.8,
                      3.0, 3.2, 3.4, 3.6, 3.8,
                      4.0, 4.2, 4.4, 4.6, 4.8,
                      5.0, 5.2, 5.4, 5.6, 5.8};
    
    // Test convection_u for a specific (i, j) point
    int i = 2;  // X-index
    int j = 2;  // Y-index
    int i_offset = 1;
    int j_offset = -1;

    Discretization d_disc(imax, jmax, dx, dy,
                        gamma);
    
    DiscretizationHost h_disc(imax, jmax, dx, dy,
                        gamma);

    double uResult = DiscretizationHost::interpolate(h_U, i, j, i_offset, j_offset);

    // Device pointers
    double *d_U, *d_uResult;
    
    // Allocate device memory
    cudaMalloc((void**)&d_U, 25 * sizeof(double));
    cudaMalloc((void**)&d_uResult, sizeof(double));
    
    // Copy data from host to device
    cudaMemcpy(d_U, h_U, 25 * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel to compute convection_u
    DiscretizationTests::interpolateTest<<<1, 1>>>(d_U, d_uResult, i, j, i_offset, j_offset, imax);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back to host
    double h_uResult;
    cudaMemcpy(&h_uResult, d_uResult, sizeof(double), cudaMemcpyDeviceToHost);
    
    EXPECT_NEAR(h_uResult, uResult, 1e-8);
}

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}