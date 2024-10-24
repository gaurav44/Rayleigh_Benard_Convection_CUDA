#include "block_sizes.hpp"
#include "cuda_utils.hpp"
#include "right_hand_side_kernels.hpp"
#include <thread>
#include <thrust/device_vector.h>

namespace RightHandSideKernels {
//__global__ void RS_kernel_call(const double *F, const double *G, double *RS,
//                               double dx, double dy, int imax, double jmax,
//                               double dt) {
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
//    int idx = imax * j + i;
//    int idxLeft = imax * j + i - 1;
//    int idxBottom = imax * (j - 1) + i;
//
//    double term1 = (F[idx] - F[idxLeft]) / dx;
//    double term2 = (G[idx] - G[idxBottom]) / dy;
//    RS[idx] = (term1 + term2) / dt;
//  }
//}

__global__ void rightHandSideKernelShared(const double *F, const double *G,
                                     double *RS, double dx, double dy, int imax,
                                     int jmax, double dt) {
  // indices offset by 1 to account for halos
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int global_idx = j * imax + i;
  __shared__ double shared_F[(BLOCK_SIZE_RS + 2) * (BLOCK_SIZE_RS + 2)];
  __shared__ double shared_G[(BLOCK_SIZE_RS + 2) * (BLOCK_SIZE_RS + 2)];

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

  // load the central part into shared memory
  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1) {
    shared_F[local_idx] = F[global_idx];
    shared_G[local_idx] = G[global_idx];
  }

  // Left Halo
  if (threadIdx.x == 0 && i > 0) {
    shared_F[local_idx - 1] = F[global_idx - 1];
  }

  // Bottom Halo
  if (threadIdx.y == 0 && j > 0) {
    shared_G[local_idx - blockDim.x - 2] = G[global_idx - imax];
  }

  __syncthreads();

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
    double term1 = (shared_F[local_idx] - shared_F[local_idx - 1]) / dx;
    double term2 =
        (shared_G[local_idx] - shared_G[local_idx - blockDim.x - 2]) / dy;
    RS[global_idx] = (term1 + term2) / dt;
  }
}

void calculateRightHandSideKernel(const Matrix &F, const Matrix &G, Matrix &RS,
               const Domain &domain) {
  dim3 threadsPerBlock(BLOCK_SIZE_RS, BLOCK_SIZE_RS);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  size_t shared_mem =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 2 * sizeof(double);

  rightHandSideKernelShared<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(F.d_container.data()),
      thrust::raw_pointer_cast(G.d_container.data()),
      thrust::raw_pointer_cast(RS.d_container.data()), domain.dx, domain.dy,
      domain.imax + 2, domain.jmax + 2, domain.dt);
  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
}
} // namespace RightHandSideKernels
