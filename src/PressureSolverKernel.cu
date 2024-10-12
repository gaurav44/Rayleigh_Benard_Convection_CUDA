#include "PressureSolver.hpp"

__global__ void SOR_kernel_call(double *P, const double *RS, double dx,
                                double dy, int imax, double jmax, double omg,
                                int color) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double coeff = omg / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)));

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1 && (i + j) % 2 == color) {
    int idx = imax * j + i;
    P[idx] =
        (1.0 - omg) * P[idx] +
        coeff * (Discretization::sor_helper(P, dx, dy, i, j, imax) - RS[idx]);
  }
}

double PressureSolver_kernel(Matrix &P, const Matrix &RS, const Domain &domain, double omg) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  double *d_P = thrust::raw_pointer_cast(P.d_container.data());
  const double *d_RS = thrust::raw_pointer_cast(RS.d_container.data());

  SOR_kernel_call<<<numBlocks, threadsPerBlock>>>(d_P, d_RS, domain.dx,
                                                  domain.dy, domain.imax + 2,
                                                  domain.jmax + 2, omg, 0);
  cudaDeviceSynchronize();

  SOR_kernel_call<<<numBlocks, threadsPerBlock>>>(d_P, d_RS, domain.dx,
                                                  domain.dy, domain.imax + 2,
                                                  domain.jmax + 2, omg, 1);
  cudaDeviceSynchronize();
  
  P.copyToHost();
  double res = 0.0;
  double rloc = 0.0;

  // Using squared value of difference to calculate residual
  for (int i = 1; i < domain.imax + 1; i++) {
    for (int j = 1; j < domain.jmax + 1; j++) {
      double val = Discretization::laplacian(P, domain, i, j) - RS(i, j);
      rloc += (val * val);
    }
  }
  res = rloc / (domain.imax * domain.jmax);
  res = std::sqrt(res);
  return res;
}
