#include "Simulation.hpp"
#include "cuda_utils.hpp"
#include <thrust/device_vector.h>

__global__ void RS_kernel_call(const double *F, const double *G, double *RS,
                               double dx, double dy, int imax, double jmax,
                               double nu, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
    int idx = imax * j + i;
    int idxLeft = imax * j + i - 1;
    int idxBottom = imax * (j - 1) + i;

    double term1 = (F[idx] - F[idxLeft]) / dx;
    double term2 = (G[idx] - G[idxBottom]) / dy;
    RS[idx] = (term1 + term2) / dt;
  }
}

void RS_kernel(const Matrix &F, const Matrix &G, Matrix &RS,
               const Domain &domain) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  RS_kernel_call<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(F.d_container.data()),
      thrust::raw_pointer_cast(G.d_container.data()),
      thrust::raw_pointer_cast(RS.d_container.data()), domain.dx, domain.dy,
      domain.imax + 2, domain.jmax + 2, domain.nu, domain.dt);
  CHECK(cudaGetLastError());
  //cudaDeviceSynchronize();
}
