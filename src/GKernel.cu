#include "Simulation.hpp"
#include <thrust/device_vector.h>
#include "cuda_utils.hpp"

__global__ void G_kernel_call(const double *U, const double *V, const double *T,
                              double *G, int imax, double jmax, double nu,
                              double dt, double GY, double beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 2) {
    int idx = imax * j + i;
    int idxTop = imax * (j + 1) + i;
    G[idx] = V[idx] +
             dt * (nu * Discretization::diffusion(V, i, j) -
                   Discretization::convection_v(U, V, i, j)) -
             (beta * dt / 2 * (T[idx] + T[idxTop])) * GY;
  }
}

void G_kernel(const Matrix &U, const Matrix &V, const Matrix &T, Matrix &G,
              const Domain &domain) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  G_kernel_call<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(U.d_container.data()),
      thrust::raw_pointer_cast(V.d_container.data()),
      thrust::raw_pointer_cast(T.d_container.data()),
      thrust::raw_pointer_cast(G.d_container.data()), domain.imax + 2,
      domain.jmax + 2, domain.nu, domain.dt, domain.GY, domain.beta);
  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
}
