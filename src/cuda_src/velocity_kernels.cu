#include "block_sizes.hpp"
#include "cuda_utils.hpp"
#include "thrust/device_vector.h"
#include "velocity_kernels.hpp"

namespace VelocityKernels {
//__global__ void U_kernel_call(double *U, const double *F, const double *P,
//                              double dx, int imax, double jmax, double dt) {
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//  if (i > 0 && j > 0 && i < imax - 2 && j < jmax - 1) {
//    int idx = imax * j + i;
//    int idxRight = imax * j + i + 1;
//    U[idx] = F[idx] - dt * (P[idxRight] - P[idx]) / dx;
//  }
//}

__global__ void U_kernelShared_call(double *U, const double *F, const double *P,
                                    double dx, int imax, double jmax,
                                    double dt) {
  // indices offset by 1 to account for halos
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int global_idx = j * imax + i;
  __shared__ double shared_P[(BLOCK_SIZE_UV + 2) * (BLOCK_SIZE_UV + 2)];

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

  // load the central part into shared memory
  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1) {
    shared_P[local_idx] = P[global_idx];
  }

  // Right Halo
  if ((threadIdx.x == blockDim.x - 1 || blockIdx.x == gridDim.x - 1) &&
      i < imax - 1) {
    shared_P[local_idx + 1] = P[global_idx + 1];
  }

  __syncthreads();

  if (i < imax - 2 && j < jmax - 1) {
    U[global_idx] = F[global_idx] -
                    dt * (shared_P[local_idx + 1] - shared_P[local_idx]) / dx;
  }
}

void U_kernel(Matrix &U, const Matrix &F, const Matrix &P,
              const Domain &domain) {
  dim3 threadsPerBlock(BLOCK_SIZE_UV, BLOCK_SIZE_UV);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  size_t shared_mem =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 1 * sizeof(double);

  U_kernelShared_call<<<numBlocks, threadsPerBlock, shared_mem>>>(
      thrust::raw_pointer_cast(U.d_container.data()),
      thrust::raw_pointer_cast(F.d_container.data()),
      thrust::raw_pointer_cast(P.d_container.data()), domain.dx,
      domain.imax + 2, domain.jmax + 2, domain.dt);

  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
}

//__global__ void V_kernel_call(double *V, const double *G, const double *P,
//                              double dy, int imax, double jmax, double dt) {
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 2) {
//    int idx = imax * j + i;
//    int idxTop = imax * (j + 1) + i;
//    V[idx] = G[idx] - dt * (P[idxTop] - P[idx]) / dy;
//  }
//}

__global__ void V_kernelShared_call(double *V, const double *G, const double *P,
                                    double dy, int imax, double jmax,
                                    double dt) {
  // indices offset by 1 to account for halos
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int global_idx = j * imax + i;
  __shared__ double shared_P[(BLOCK_SIZE_UV + 2) * (BLOCK_SIZE_UV + 2)];
  ;

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

  // load the central part into shared memory
  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1) {
    shared_P[local_idx] = P[global_idx];
  }

  // Top Halo
  if ((threadIdx.y == blockDim.y - 1 || blockIdx.y == gridDim.y - 1) &&
      j < jmax - 1) {
    shared_P[local_idx + blockDim.x + 2] = P[global_idx + imax];
  }

  __syncthreads();

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 2) {
    V[global_idx] =
        G[global_idx] -
        dt * (shared_P[local_idx + blockDim.x + 2] - shared_P[local_idx]) / dy;
  }
}

void V_kernel(Matrix &V, const Matrix &G, const Matrix &P,
              const Domain &domain) {
  dim3 threadsPerBlock(BLOCK_SIZE_UV, BLOCK_SIZE_UV);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  size_t shared_mem =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 1 * sizeof(double);

  V_kernelShared_call<<<numBlocks, threadsPerBlock, shared_mem>>>(
      thrust::raw_pointer_cast(V.d_container.data()),
      thrust::raw_pointer_cast(G.d_container.data()),
      thrust::raw_pointer_cast(P.d_container.data()), domain.dy,
      domain.imax + 2, domain.jmax + 2, domain.dt);

  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
}

__global__ void velocityKernelShared(double *U, double *V, const double *F,
                                     const double *G, const double *P,
                                     double dx, double dy, int imax,
                                     double jmax, double dt) {
  // indices offset by 1 to account for halos
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int global_idx = j * imax + i;
  __shared__ double shared_P[(BLOCK_SIZE_UV + 2) * (BLOCK_SIZE_UV + 2)];

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

  // load the central part into shared memory
  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1) {
    shared_P[local_idx] = P[global_idx];
  }

  // Right Halo
  if (threadIdx.x == blockDim.x - 1 && i < imax - 1) {
    shared_P[local_idx + 1] = P[global_idx + 1];
  }

  // Top Halo
  if (threadIdx.y == blockDim.y - 1 && j < jmax - 1) {
    shared_P[local_idx + blockDim.x + 2] = P[global_idx + imax];
  }

  __syncthreads();

  if (i < imax - 1 && j < jmax - 2) {
    V[global_idx] =
        G[global_idx] -
        dt * (shared_P[local_idx + blockDim.x + 2] - shared_P[local_idx]) / dy;
  }

  if (i < imax - 2 && j < jmax - 1) {
    U[global_idx] = F[global_idx] -
                    dt * (shared_P[local_idx + 1] - shared_P[local_idx]) / dx;
  }
}

void calculateVelocitiesKernel(Matrix &U, Matrix &V, const Matrix &F,
                             const Matrix &G, const Matrix &P,
                             const Domain &domain) {
  dim3 threadsPerBlock(BLOCK_SIZE_UV, BLOCK_SIZE_UV);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  size_t shared_mem =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 1 * sizeof(double);

  velocityKernelShared<<<numBlocks, threadsPerBlock, shared_mem>>>(
      thrust::raw_pointer_cast(U.d_container.data()),
      thrust::raw_pointer_cast(V.d_container.data()),
      thrust::raw_pointer_cast(F.d_container.data()),
      thrust::raw_pointer_cast(G.d_container.data()),
      thrust::raw_pointer_cast(P.d_container.data()), domain.dx, domain.dy,
      domain.imax + 2, domain.jmax + 2, domain.dt);
  CHECK(cudaGetLastError());
}
} // namespace VelocityKernels
