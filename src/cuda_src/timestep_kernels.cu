#include "block_sizes.hpp"
#include "cuda_utils.hpp"
#include "timestep_kernels.hpp"
#include <cmath>
#include <iterator>
#include <thrust/device_vector.h>

namespace TimestepKernels {
__global__ void velocityUMaxKernel(const double *U, int imax, int jmax,
                                 double *max_results) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ double shared_data[BLOCK_SIZE_DT * BLOCK_SIZE_DT];
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

__global__ void velocityVMaxKernel(const double *V, int imax, int jmax,
                                 double *max_results) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ double shared_data[BLOCK_SIZE_DT * BLOCK_SIZE_DT];
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

std::pair<double, double> calculateTimeStepKernel(const Matrix &U, const Matrix &V,
                                    const Domain &domain, double *d_uBlockMax,
                                    double *d_vBlockMax,
                                    double *h_uBlockMax,
                                    double *h_vBlockMax) {
  dim3 threadsPerBlock(BLOCK_SIZE_DT, BLOCK_SIZE_DT);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  double h_uMax = 0.0;
  double h_vMax = 0.0;

  velocityUMaxKernel<<<numBlocks, threadsPerBlock,
                     BLOCK_SIZE_DT * BLOCK_SIZE_DT * sizeof(double)>>>(
      thrust::raw_pointer_cast(U.d_container.data()), domain.imax + 2,
      domain.jmax + 2, d_uBlockMax);
  CHECK(cudaGetLastError());

  velocityVMaxKernel<<<numBlocks, threadsPerBlock,
                     BLOCK_SIZE_DT * BLOCK_SIZE_DT * sizeof(double)>>>(
      thrust::raw_pointer_cast(V.d_container.data()), domain.imax + 2,
      domain.jmax + 2, d_vBlockMax);
  CHECK(cudaGetLastError());

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
  return {h_uMax, h_vMax};
}
} // namespace TimestepKernels
