#include "block_sizes.hpp"
#include "cuda_utils.hpp"
#include "discretization.hpp"
#include "temperature_kernels.hpp"
#include <thread>
#include <thrust/device_vector.h>

namespace TemperatureKernels {
//__global__ void temperature_kernel_call(const double *U, const double *V,
//                                        double *T, const double *T_old,
//                                        double dx, double dy, int imax,
//                                        double jmax, double gamma, double
//                                        alpha, double dt) {
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
//    int idx = j * imax + i;
//    T[idx] =
//        T_old[idx] + dt * (alpha * Discretization::diffusion(T_old, i, j) -
//                           Discretization::convection_T(U, V, T_old, i, j));
//  }
//}

__global__ void temperatureKernelShared(const double *U, const double *V,
                                        double *T, int imax, int jmax,
                                        double alpha, double dt) {
  // indices offset by 1 to account for halos
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int global_idx = j * imax + i;
  __shared__ double shared_Told[(BLOCK_SIZE_TEMP + 2) * (BLOCK_SIZE_TEMP + 2)];
  __shared__ double shared_U[(BLOCK_SIZE_TEMP + 2) * (BLOCK_SIZE_TEMP + 2)];
  __shared__ double shared_V[(BLOCK_SIZE_TEMP + 2) * (BLOCK_SIZE_TEMP + 2)];

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

  // load the central part into shared memory
  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1) {
    shared_Told[local_idx] = T[global_idx];
    shared_U[local_idx] = U[global_idx];
    shared_V[local_idx] = V[global_idx];
  }

  // Left Halo
  if (threadIdx.x == 0 && i > 0) {
    shared_Told[local_idx - 1] = T[global_idx - 1];
    shared_U[local_idx - 1] = U[global_idx - 1];
    shared_V[local_idx - 1] = V[global_idx - 1];
  }

  // Right Halo
  if ((threadIdx.x == blockDim.x - 1 ||
       (blockIdx.x == gridDim.x - 1 &&
        threadIdx.x == (imax - 2) % blockDim.x)) &&
      i < imax - 1) {
    shared_Told[local_idx + 1] = T[global_idx + 1];
    shared_U[local_idx + 1] = U[global_idx + 1];
    shared_V[local_idx + 1] = V[global_idx + 1];
  }

  // Bottom Halo
  if (threadIdx.y == 0 && j > 0) {
    shared_Told[local_idx - blockDim.x - 2] = T[global_idx - imax];
    shared_U[local_idx - blockDim.x - 2] = U[global_idx - imax];
    shared_V[local_idx - blockDim.x - 2] = V[global_idx - imax];
  }

  // Top Halo
  if ((threadIdx.y == blockDim.y - 1 ||
      (blockIdx.y == gridDim.y - 1) &&
       threadIdx.y == (jmax - 2) % blockDim.y) &&
      j < jmax - 1) {
    shared_Told[local_idx + blockDim.x + 2] = T[global_idx + imax];
    shared_U[local_idx + blockDim.x + 2] = U[global_idx + imax];
    shared_V[local_idx + blockDim.x + 2] = V[global_idx + imax];
  }

  __syncthreads();

  if (i < imax - 1 && j < jmax - 1) {
    T[global_idx] =
        shared_Told[local_idx] +
        dt * (alpha * Discretization::diffusionSharedMem(
                          shared_Told, local_i, local_j, blockDim.x + 2) -
              Discretization::convection_TSharedMem(shared_U, shared_V,
                                                    shared_Told, local_i,
                                                    local_j, blockDim.x + 2));
  }
}

void calculateTemperatureKernel(const Matrix &U, const Matrix &V, Matrix &T,
                                const Domain &domain) {

  dim3 threadsPerBlock(BLOCK_SIZE_TEMP, BLOCK_SIZE_TEMP);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  size_t shared_mem =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 3 * sizeof(double);

  temperatureKernelShared<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(U.d_container.data()),
      thrust::raw_pointer_cast(V.d_container.data()),
      thrust::raw_pointer_cast(T.d_container.data()), domain.imax + 2,
      domain.jmax + 2, domain.alpha, domain.dt);

  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
}
} // namespace TemperatureKernels
