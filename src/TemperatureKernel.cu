#include "Simulation.hpp"
#include "cuda_utils.hpp"
#include <thread>
#include <thrust/device_vector.h>

#define BLOCK_SIZE 16

__global__ void temperature_kernel_call(const double *U, const double *V,
                                        double *T, const double *T_old,
                                        double dx, double dy, int imax,
                                        double jmax, double gamma, double alpha,
                                        double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
    int idx = j * imax + i;
    T[idx] =
        T_old[idx] + dt * (alpha * Discretization::diffusion(T_old, i, j) -
                           Discretization::convection_T(U, V, T_old, i, j));
  }
}

__global__ void temperature_kernelShared_call(const double *U, const double *V,
                                              double *T, const double *T_old,
                                              double dx, double dy, int imax,
                                              double jmax, double gamma,
                                              double alpha, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int global_idx = j * imax + i;
  extern __shared__ double buffer[];

  double *shared_Told = &buffer[0];
  double *shared_T = &buffer[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)];
  double *shared_U = &buffer[2 * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)];
  double *shared_V = &buffer[3 * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)];

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

  // if(local_idx == 34) printf("global_idx:%d, threadIdx.x:%d. threadIdx.y:%d,
  // local_idx:%d\n", global_idx, threadIdx.x, threadIdx.y, local_idx);

  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1) {
    shared_T[local_idx] = T[global_idx];
    shared_Told[local_idx] = T_old[global_idx];
    shared_U[local_idx] = U[global_idx];
    shared_V[local_idx] = V[global_idx];
  }
  // if (i == 6 && j == 1) {

  //  printf("*******************************************************************"
  //         "**************\n");
  //  for (int jj = 1; jj < blockDim.y + 1; jj++) {
  //    for (int ii = 1; ii < blockDim.x + 1; ii++) {
  //      printf(" %.2f ", shared_Told[jj * (blockDim.x + 2) + ii]);
  //    }
  //    printf("\n");
  //  }

  //  printf("*******************************************************************"
  //         "**************\n");
  //  for (int j = 1; j < jmax - 1; j++) {
  //    for (int i = 1; i < imax - 1; i++) {
  //      printf(" %.2f ", T_old[j * imax + i]);
  //    }
  //    printf("\n");
  //  }

  //  printf("*******************************************************************"
  //         "**************\n");
  //}

  // Left Halo
  if (threadIdx.x == 0 && i > 0) {
    shared_T[local_idx - 1] = T[global_idx - 1];
    shared_Told[local_idx - 1] = T_old[global_idx - 1];
    shared_U[local_idx - 1] = U[global_idx - 1];
    shared_V[local_idx - 1] = V[global_idx - 1];
  }

  // Right Halo
  if (threadIdx.x == blockDim.x - 1 && i < imax - 1) {
    shared_T[local_idx + 1] = T[global_idx + 1];
    shared_Told[local_idx + 1] = T_old[global_idx + 1];
    shared_U[local_idx + 1] = U[global_idx + 1];
    shared_V[local_idx + 1] = V[global_idx + 1];
  }

  // Bottom Halo
  if (threadIdx.y == 0 && j > 0) {
    shared_T[local_idx - blockDim.x - 2] = T[global_idx - imax];
    shared_Told[local_idx - blockDim.x - 2] = T_old[global_idx - imax];
    shared_U[local_idx - blockDim.x - 2] = U[global_idx - imax];
    shared_V[local_idx - blockDim.x - 2] = V[global_idx - imax];
  }

  // Top Halo
  if (threadIdx.y == blockDim.y - 1 && j < jmax - 1) {
    shared_T[local_idx + blockDim.x + 2] = T[global_idx + imax];
    shared_Told[local_idx + blockDim.x + 2] = T_old[global_idx + imax];
    shared_U[local_idx + blockDim.x + 2] = U[global_idx + imax];
    shared_V[local_idx + blockDim.x + 2] = V[global_idx + imax];
  }

  __syncthreads();
  // printf("Before Update: T_old[%d] = %f, U[%d] = %f, V[%d] = %f\n",
  // global_idx,
  //        T_old[global_idx], global_idx, U[global_idx], global_idx,
  //        V[global_idx]);
  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {

   // if (fabs(T_old[global_idx + imax] -
   //          shared_Told[local_idx + blockDim.x + 2]) > 1e-8)
   //   printf("i:%d, j:%d, local_i:%d, local_j:%d, T_old= %f, "
   //          "shared_Told= %f\n",
   //          i - 1, j, local_i - 1, local_j, T_old[global_idx - 1],
   //          shared_Told[local_idx - 1]);

    shared_T[local_idx] =
        shared_Told[local_idx] +
        dt * (alpha * Discretization::diffusionSharedMem(
                          shared_Told, local_i, local_j, blockDim.x + 2) -
              Discretization::convection_T(U, V, T_old, i, j));
    T[global_idx] = shared_T[local_idx]; //(threadIdx.y+1)*(blockDim.x+2) +
  }

  // if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
  //   T[global_idx] = shared_T[local_idx]; //(threadIdx.y+1)*(blockDim.x+2) +
  //        threadIdx.x+1];
  // }
}

void temperature_kernel(const Matrix &U, const Matrix &V, Matrix &T,
                        const Domain &domain) {

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  thrust::device_vector<double> T_old(T.d_container.size());
  thrust::copy(T.d_container.begin(), T.d_container.end(), T_old.begin());

  size_t shared_mem =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 4 * sizeof(double);
  // int maxSharedMemSize;
  // cudaDeviceGetAttribute(&maxSharedMemSize,
  // cudaDevAttrMaxSharedMemoryPerBlock,
  //                       0);
  temperature_kernelShared_call<<<numBlocks, threadsPerBlock, shared_mem>>>(
      thrust::raw_pointer_cast(U.d_container.data()),
      thrust::raw_pointer_cast(V.d_container.data()),
      thrust::raw_pointer_cast(T.d_container.data()),
      thrust::raw_pointer_cast(T_old.data()), domain.dx, domain.dy,
      domain.imax + 2, domain.jmax + 2, domain.gamma, domain.alpha, domain.dt);

  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
}
