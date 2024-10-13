#include "Simulation.hpp"
#include <thrust/device_vector.h>

__global__ void F_kernel_call(const double *U, const double *V, const double *T,
                              double *F, int imax, int jmax, double nu, double dt,
                              double GX, double beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < imax - 2 && j < jmax - 1) {
    int idx = imax * j + i;
    int idxRight = imax * j + i + 1;
    F[idx] = U[idx] +
             dt * (nu * Discretization::diffusion(U, i, j) -
                   Discretization::convection_u(U, V, i, j)) -
             (beta * dt / 2 * (T[idx] + T[idxRight])) * GX;
  }
}

void F_kernel(const Matrix &U, const Matrix &V, const Matrix &T, Matrix &F,
              const Domain &domain) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  F_kernel_call<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(U.d_container.data()),
      thrust::raw_pointer_cast(V.d_container.data()),
      thrust::raw_pointer_cast(T.d_container.data()),
      thrust::raw_pointer_cast(F.d_container.data()), domain.imax + 2,
      domain.jmax + 2, domain.nu, domain.dt, domain.GX, domain.beta);
  // cudaDeviceSynchronize();
}
