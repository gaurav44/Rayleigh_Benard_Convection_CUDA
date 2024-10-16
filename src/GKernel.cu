#include "Simulation.hpp"
#include "cuda_utils.hpp"
#include <thrust/device_vector.h>

#define BLOCK_SIZE 16

//__global__ void G_kernel_call(const double *U, const double *V, const double *T,
//                              double *G, int imax, double jmax, double nu,
//                              double dt, double GY, double beta) {
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 2) {
//    int idx = imax * j + i;
//    int idxTop = imax * (j + 1) + i;
//    G[idx] = V[idx] +
//             dt * (nu * Discretization::diffusion(V, i, j) -
//                   Discretization::convection_v(U, V, i, j)) -
//             (beta * dt / 2 * (T[idx] + T[idxTop])) * GY;
//  }
//}

__global__ void G_kernelShared_call(const double *U, const double *V,
                                    const double *T, double *G, int imax,
                                    double jmax, double nu, double dt,
                                    double GY, double beta) {
  // indices offset by 1 to account for halos
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int global_idx = j * imax + i;
  __shared__ double shared_U[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)];
  __shared__ double shared_V[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)];
  __shared__ double shared_T[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)];

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

  // load the central part into shared memory
  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1) {
    shared_U[local_idx] = U[global_idx];
    shared_V[local_idx] = V[global_idx];
    shared_T[local_idx] = T[global_idx];
  }

  // Left Halo
  if (threadIdx.x == 0 && i > 0) {
    shared_U[local_idx - 1] = U[global_idx - 1];
    shared_V[local_idx - 1] = V[global_idx - 1];
  }

  // Right Halo
  if (threadIdx.x == blockDim.x - 1 && i < imax - 1) {
    shared_U[local_idx + 1] = U[global_idx + 1];
    shared_V[local_idx + 1] = V[global_idx + 1];
  }

  // Bottom Halo
  if (threadIdx.y == 0 && j > 0) {
    shared_U[local_idx - blockDim.x - 2] = U[global_idx - imax];
    shared_V[local_idx - blockDim.x - 2] = V[global_idx - imax];
  }

  // Top Halo
  if (threadIdx.y == blockDim.y - 1 && j < jmax - 1) {
    shared_U[local_idx + blockDim.x + 2] = U[global_idx + imax];
    shared_U[local_idx + blockDim.x + 2 - 1] = U[global_idx + imax - 1];
    shared_V[local_idx + blockDim.x + 2] = V[global_idx + imax];
    shared_T[local_idx + blockDim.x + 2] = T[global_idx + imax];
  }

  __syncthreads();

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 2) {
    G[global_idx] =
        shared_V[local_idx] +
        dt * (nu * Discretization::diffusionSharedMem(shared_V, local_i,
                                                      local_j, blockDim.x + 2) -
              Discretization::convection_vSharedMem(shared_U, shared_V, local_i,
                                                    local_j, blockDim.x + 2)) -
        (0.5 * beta * dt *
         (shared_T[local_idx] + shared_T[local_idx + blockDim.x + 2])) *
            GY;
  }
}

void G_kernel(const Matrix &U, const Matrix &V, const Matrix &T, Matrix &G,
              const Domain &domain) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  size_t shared_mem =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 3 * sizeof(double);

  G_kernelShared_call<<<numBlocks, threadsPerBlock, shared_mem>>>(
      thrust::raw_pointer_cast(U.d_container.data()),
      thrust::raw_pointer_cast(V.d_container.data()),
      thrust::raw_pointer_cast(T.d_container.data()),
      thrust::raw_pointer_cast(G.d_container.data()), domain.imax + 2,
      domain.jmax + 2, domain.nu, domain.dt, domain.GY, domain.beta);
  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
}
