#include "Simulation.hpp"
#include "thrust/device_vector.h"

__global__ void U_kernel_call(double *U, const double *F, const double *P,
                              double dx, int imax, double jmax, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < imax - 2 && j < jmax - 1) {
    int idx = imax * j + i;
    int idxRight = imax * j + i + 1;
    U[idx] = F[idx] - dt * (P[idxRight] - P[idx]) / dx;
  }
}

void U_kernel(Matrix &U, const Matrix &F, const Matrix &P,
              const Domain &domain) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  U_kernel_call<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(U.d_container.data()),
      thrust::raw_pointer_cast(F.d_container.data()),
      thrust::raw_pointer_cast(P.d_container.data()), domain.dx,
      domain.imax + 2, domain.jmax + 2, domain.dt);
  // cudaDeviceSynchronize();
}

__global__ void V_kernel_call(double *V, const double *G, const double *P,
                              double dy, int imax, double jmax, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 2) {
    int idx = imax * j + i;
    int idxTop = imax * (j + 1) + i;
    V[idx] = G[idx] - dt * (P[idxTop] - P[idx]) / dy;
  }
}

void V_kernel(Matrix &V, const Matrix &G, const Matrix &P,
              const Domain &domain) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  V_kernel_call<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(V.d_container.data()),
      thrust::raw_pointer_cast(G.d_container.data()),
      thrust::raw_pointer_cast(P.d_container.data()), domain.dy,
      domain.imax + 2, domain.jmax + 2, domain.dt);
  // cudaDeviceSynchronize();
}
