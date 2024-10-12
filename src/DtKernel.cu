#include "Simulation.hpp"
#include <cmath>
#include <iterator>
#include <thrust/device_vector.h>

__global__ void Max_kernel_call(const double *U, const double *V, int imax,
                                double jmax, double *u_max,
                                double *v_max) { // double *max_results) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // extern __shared__ double shared_data[];
  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
    int idx = j * imax + i;
    if (i == 1 && j == 1) {
      for (int ii = 1; ii < imax - 1; ii++) {
        for (int jj = 1; jj < jmax - 1; jj++) {
          *u_max = fmax(*u_max, fabs(U[idx]));
          *v_max = fmax(*v_max, fabs(V[idx]));
        }
      }
    }
    //  int idx = imax * j + i;

    //  int shared_index = threadIdx.x * blockDim.y + threadIdx.y;
    //  shared_data[shared_index] = 0.0;
    //  shared_data[shared_index] = fabs(U[idx]);
    //  __syncthreads(); // Synchronize to ensure all data is loaded

    //  // Perform reduction in shared memory
    //  for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s /= 2) {
    //    if (shared_index < s) {
    //      shared_data[shared_index] =
    //          fmax(shared_data[shared_index], shared_data[shared_index + s]);
    //    }
    //    __syncthreads(); // Synchronize after each reduction step
    //  }
    //}
    //// Write the result of this block's max to global memory
    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //   max_results[blockIdx.x * gridDim.y + blockIdx.y] = shared_data[0];
  }
}

__global__ void VMax_kernel_call(const double *V, int imax, double jmax,
                                 double *max_results) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  extern __shared__ double shared_data[];
  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
    int idx = imax * j + i;

    int shared_index = threadIdx.x * blockDim.y + threadIdx.y;
    shared_data[shared_index] = 0.0;
    shared_data[shared_index] = fabs(V[idx]);
    __syncthreads(); // Synchronize to ensure all data is loaded

    // Perform reduction in shared memory
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s /= 2) {
      if (shared_index < s) {
        shared_data[shared_index] =
            fmax(shared_data[shared_index], shared_data[shared_index + s]);
      }
      __syncthreads(); // Synchronize after each reduction step
    }
  }
  // Write the result of this block's max to global memory
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    max_results[blockIdx.x * gridDim.y + blockIdx.y] = shared_data[0];
  }
}

std::pair<double, double> Dt_kernel(const Matrix &U, const Matrix &V,
                                    const Domain &domain) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  const double *d_U = thrust::raw_pointer_cast(U.d_container.data());
  const double *d_V = thrust::raw_pointer_cast(V.d_container.data());

  double h_u_max = 0.0;
  double h_v_max = 0.0;

  double *d_u_max;
  double *d_v_max;
  cudaMalloc(&d_u_max, sizeof(double));
  cudaMalloc(&d_v_max, sizeof(double));

  cudaMemcpy(d_u_max, &h_u_max, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_max, &h_v_max, sizeof(double), cudaMemcpyHostToDevice);

  // double *d_u_block_results;
  // double *d_v_block_results;

  // cudaMalloc(&d_u_block_results, 256 * sizeof(double));
  // cudaMalloc(&d_v_block_results, 256 * sizeof(double));

  Max_kernel_call<<<numBlocks, threadsPerBlock, 256 * sizeof(double)>>>(
      d_U, d_V, domain.imax + 2, domain.jmax + 2, d_u_max, d_v_max);
  // VMax_kernel_call<<<numBlocks, threadsPerBlock, 256 * sizeof(double)>>>(
  //     d_V, domain.imax + 2, domain.jmax + 2, d_v_block_results);
  // cudaDeviceSynchronize();
  cudaMemcpy(&h_u_max, d_u_max, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_v_max, d_v_max, sizeof(double), cudaMemcpyDeviceToHost);
  //double *h_umax_results = new double[256];
  //double *h_vmax_results = new double[256];
  //cudaMemcpy(h_umax_results, d_u_block_results, 256 * sizeof(double),
  //           cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_vmax_results, d_v_block_results, 256 * sizeof(double),
  //           cudaMemcpyDeviceToHost);

  // Find the maximum in the result array
  // for (int i = 0; i < 256; ++i) {
  //  h_u_max = fmax(h_u_max, h_umax_results[i]);
  //  h_v_max = fmax(h_v_max, h_vmax_results[i]);
  //}

  return {h_u_max, h_v_max};
}
