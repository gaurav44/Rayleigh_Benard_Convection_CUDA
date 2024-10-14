#include "PressureSolver.hpp"
#include "cuda_utils.hpp"

__global__ void SOR_kernel_call(double *P, const double *RS, double dx,
                                double dy, int imax, double jmax, double omg,
                                int color) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double coeff = omg / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)));

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1 && (i + j) % 2 == color) {
    int idx = imax * j + i;
    P[idx] = (1.0 - omg) * P[idx] +
             coeff * (Discretization::sor_helper(P, i, j) - RS[idx]);
  }
}

__global__ void Residual_kernel_call(const double *P, const double *RS,
                                     int imax, double jmax, double *residual) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
    int idx = imax * j + i;
    double val = Discretization::laplacian(P, i, j) - RS[idx];
    atomicAdd(residual, (val * val));
  }
}

double PressureSolver_kernel(Matrix &P, const Matrix &RS, const Domain &domain,
                             double omg) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  double *d_P = thrust::raw_pointer_cast(P.d_container.data());
  const double *d_RS = thrust::raw_pointer_cast(RS.d_container.data());

  SOR_kernel_call<<<numBlocks, threadsPerBlock>>>(d_P, d_RS, domain.dx,
                                                  domain.dy, domain.imax + 2,
                                                  domain.jmax + 2, omg, 0);

  CHECK(cudaGetLastError());

  SOR_kernel_call<<<numBlocks, threadsPerBlock>>>(d_P, d_RS, domain.dx,
                                                  domain.dy, domain.imax + 2,
                                                  domain.jmax + 2, omg, 1);
  CHECK(cudaGetLastError());

  double res = 0.0;
  double *d_rloc;
  double h_rloc = 0.0;
  CHECK(cudaMalloc(&d_rloc, sizeof(double)));
  CHECK(cudaMemcpy(d_rloc, &h_rloc, sizeof(double), cudaMemcpyHostToDevice));

  Residual_kernel_call<<<numBlocks, threadsPerBlock>>>(
      d_P, d_RS, domain.imax + 2, domain.jmax + 2, d_rloc);

  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
  CHECK(cudaMemcpy(&h_rloc, d_rloc, sizeof(double), cudaMemcpyDeviceToHost));
  res = h_rloc / (domain.imax * domain.jmax);
  res = std::sqrt(res);
  CHECK(cudaFree(d_rloc));
  return res;
}
