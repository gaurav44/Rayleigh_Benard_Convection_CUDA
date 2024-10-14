#include "Simulation.hpp"
#include <thrust/device_vector.h>
#include "cuda_utils.hpp"

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

void temperature_kernel(const Matrix &U, const Matrix &V, Matrix &T,
                        const Domain &domain) {

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  thrust::device_vector<double> T_old(T.d_container.size());
  thrust::copy(T.d_container.begin(), T.d_container.end(), T_old.begin());

  temperature_kernel_call<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(U.d_container.data()),
      thrust::raw_pointer_cast(V.d_container.data()),
      thrust::raw_pointer_cast(T.d_container.data()),
      thrust::raw_pointer_cast(T_old.data()), domain.dx, domain.dy,
      domain.imax + 2, domain.jmax + 2, domain.gamma, domain.alpha, domain.dt);
  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
}
