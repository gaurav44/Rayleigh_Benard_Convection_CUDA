#include "PressureSolver.hpp"
#include "cuda_utils.hpp"

#define BLOCK_SIZE 16

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

__global__ void SOR_kernelShared_call(double *P, const double *RS, int imax,
                                      double jmax, double omg, double coeff,
                                      int color) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int global_idx = j * imax + i;
  extern __shared__ double buffer[];
  double *shared_P = &buffer[0];
  double *shared_RS = &buffer[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)];

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

 // double coeff = omg / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)));

  // load the central part into shared memory
  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1) {
    shared_P[local_idx] = P[global_idx];
    shared_RS[local_idx] = RS[global_idx];
  }

  // Left Halo
  if (threadIdx.x == 0 && i > 0) {
    shared_P[local_idx - 1] = P[global_idx - 1];
    shared_RS[local_idx - 1] = RS[global_idx - 1];
  }

  // Right Halo
  if (threadIdx.x == blockDim.x - 1 && i < imax - 1) {
    shared_P[local_idx + 1] = P[global_idx + 1];
    shared_RS[local_idx + 1] = RS[global_idx + 1];
  }

  // Bottom Halo
  if (threadIdx.y == 0 && j > 0) {
    shared_P[local_idx - blockDim.x - 2] = P[global_idx - imax];
    shared_RS[local_idx - blockDim.x - 2] = RS[global_idx - imax];
  }

  // Top Halo
  if (threadIdx.y == blockDim.y - 1 && j < jmax - 1) {
    shared_P[local_idx + blockDim.x + 2] = P[global_idx + imax];
    shared_RS[local_idx + blockDim.x + 2] = RS[global_idx + imax];
  }

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1 && (i + j) % 2 == color) {
    // int idx = imax * j + i;
    shared_P[local_idx] =
        (1.0 - omg) * shared_P[local_idx] +
        coeff * (Discretization::sor_helperSharedMem(shared_P, local_i, local_j,
                                                     blockDim.x + 2) -
                 shared_RS[local_idx]);
    P[global_idx] = shared_P[local_idx];
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

__global__ void Residual_kernelShared_call(const double *P, const double *RS,
                                           int imax, double jmax,
                                           double *residual) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
    int idx = imax * j + i;
    double val = Discretization::laplacian(P, i, j) - RS[idx];
    atomicAdd(residual, (val * val));
  }
}

double PressureSolver_kernel(Matrix &P, const Matrix &RS, const Domain &domain,
                             double omg, double *d_rloc) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  
  size_t shared_mem =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 4 * sizeof(double);

  double coeff = omg / (2.0 * (1.0 / (domain.dx * domain.dx) + 1.0 / (domain.dy * domain.dy)));
 

  SOR_kernelShared_call<<<numBlocks, threadsPerBlock, shared_mem>>>(
      thrust::raw_pointer_cast(P.d_container.data()),
      thrust::raw_pointer_cast(RS.d_container.data()), domain.imax + 2,
      domain.jmax + 2, omg, coeff, 0);

  CHECK(cudaGetLastError());

  SOR_kernelShared_call<<<numBlocks, threadsPerBlock, shared_mem>>>(
      thrust::raw_pointer_cast(P.d_container.data()),
      thrust::raw_pointer_cast(RS.d_container.data()), domain.imax + 2,
      domain.jmax + 2, omg, coeff, 1);
  CHECK(cudaGetLastError());

  double res = 0.0;
  // double *d_rloc;
  double h_rloc = 0.0;
  // CHECK(cudaMalloc(&d_rloc, sizeof(double)));
  CHECK(cudaMemcpy(d_rloc, &h_rloc, sizeof(double), cudaMemcpyHostToDevice));

  Residual_kernel_call<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(P.d_container.data()),
      thrust::raw_pointer_cast(RS.d_container.data()), domain.imax + 2,
      domain.jmax + 2, d_rloc);

  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
  CHECK(cudaMemcpy(&h_rloc, d_rloc, sizeof(double), cudaMemcpyDeviceToHost));
  res = h_rloc / (domain.imax * domain.jmax);
  res = std::sqrt(res);
  // CHECK(cudaFree(d_rloc));
  return res;
}
