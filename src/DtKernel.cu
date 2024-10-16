#include "Simulation.hpp"
#include <cmath>
#include <iterator>
#include <thrust/device_vector.h>
#include "cuda_utils.hpp"

#define BLOCK_SIZE 32

__global__ void UMax_kernel_call(const double *U, int imax, double jmax,
                                 double *max_results) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ double shared_data[BLOCK_SIZE*BLOCK_SIZE];
  int shared_index = threadIdx.x * blockDim.y + threadIdx.y;
  shared_data[shared_index] = 0.0;

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
    int idx = imax * j + i;

    shared_data[shared_index] = fabs(U[idx]);
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

__global__ void VMax_kernel_call(const double *V, int imax, double jmax,
                                 double *max_results) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ double shared_data[BLOCK_SIZE*BLOCK_SIZE];
  int shared_index = threadIdx.x * blockDim.y + threadIdx.y;
  shared_data[shared_index] = 0.0;

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
    int idx = imax * j + i;

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
                                    const Domain &domain, double* d_u_block_max, double* d_v_block_max,
                                    double* h_u_block_max, double* h_v_block_max) {
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  double h_u_max = 0.0;
  double h_v_max = 0.0;

  UMax_kernel_call<<<numBlocks, threadsPerBlock, 1024 * sizeof(double)>>>(
      thrust::raw_pointer_cast(U.d_container.data()), domain.imax + 2,
      domain.jmax + 2, d_u_block_max);
  CHECK(cudaGetLastError());

  VMax_kernel_call<<<numBlocks, threadsPerBlock, 1024 * sizeof(double)>>>(
      thrust::raw_pointer_cast(V.d_container.data()), domain.imax + 2,
      domain.jmax + 2, d_v_block_max);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(h_u_block_max, d_u_block_max,
             numBlocks.x * numBlocks.y * sizeof(double),
             cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_v_block_max, d_v_block_max,
             numBlocks.x * numBlocks.y * sizeof(double),
             cudaMemcpyDeviceToHost));

  // Find the maximum in the result array
  for (int i = 0; i < numBlocks.x * numBlocks.y; ++i) {
    h_u_max = fmax(h_u_max, h_u_block_max[i]);
    h_v_max = fmax(h_v_max, h_v_block_max[i]);
  }
  return {h_u_max, h_v_max};
}
