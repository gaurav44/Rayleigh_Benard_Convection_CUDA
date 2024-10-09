#include "Simulation.hpp"
#include <thrust/device_vector.h>

__global__ void kernel_call(double *U, double *V, double *T,
                            double *T_old, double dx, double dy,
                            double imax, double jmax, double gamma,
                            double alpha, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = i * imax + j;

  if (i > 0 && j > 0 && i < imax && j < jmax) {
    T[idx] =
        T_old[idx] +
        dt * (alpha * Discretization::diffusion(T_old, dx, dy, i, j, imax) -
              Discretization::convection_T(U, V, T_old, gamma, dx, dy, i, j,
                                           imax));
  }
  printf("cold:%d", j);
}

void temperature_kernel(Matrix &U, Matrix &V, Matrix &T,
                        Matrix &T_old, const Domain &domain) {

   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks((domain.imax + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (domain.jmax + threadsPerBlock.y - 1) / threadsPerBlock.y);

  double *d_U = thrust::raw_pointer_cast(U.d_container.data());
  double *d_V = thrust::raw_pointer_cast(V.d_container.data());
  double *d_T = thrust::raw_pointer_cast(T.d_container.data());
  double *dT_old = thrust::raw_pointer_cast(T_old.d_container.data());

  kernel_call<<<10, 16>>>(d_U, d_V, d_T, dT_old, domain.dx, domain.dy,
                          domain.imax, domain.jmax, domain.gamma, domain.alpha, domain.dt);
  cudaDeviceSynchronize();
}
