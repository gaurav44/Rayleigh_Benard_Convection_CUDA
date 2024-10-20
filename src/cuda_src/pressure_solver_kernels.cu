#include "block_sizes.hpp"
#include "cuda_utils.hpp"
#include "pressure_solver_kernels.hpp"
#include "discretization.hpp"
#include <vector>

//#define BLOCK_SIZE 16
namespace PressureSolverKernels {
__global__ void SOR_kernel_call(double *P, const double *RS, int imax,
                                double jmax, double omg, double coeff,
                                int color) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  //  double coeff = omg / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)));

  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1 && (i + j) % 2 == color) {
    int idx = imax * j + i;
    P[idx] = (1.0 - omg) * P[idx] +
             coeff * (Discretization::sor_helper(P, i, j) - RS[idx]);
  }
}

__global__ void SORKernelShared(double *P, const double *RS, int imax,
                                      double jmax, double omg, double coeff,
                                      int color) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int global_idx = j * imax + i;
  __shared__ double shared_P[(BLOCK_SIZE_SOR + 2) * (BLOCK_SIZE_SOR + 2)];

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

  // load the central part into shared memory
  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1)
    shared_P[local_idx] = P[global_idx];

  // Left Halo
  if (threadIdx.x == 0 && i > 0)
    shared_P[local_idx - 1] = P[global_idx - 1];

  // Right Halo
  if ((threadIdx.x == blockDim.x - 1 || blockIdx.x == gridDim.x - 1) &&
      i < imax - 1)
    shared_P[local_idx + 1] = P[global_idx + 1];

  // Bottom Halo
  if (threadIdx.y == 0 && j > 0)
    shared_P[local_idx - blockDim.x - 2] = P[global_idx - imax];

  // Top Halo
  if ((threadIdx.y == blockDim.y - 1 || blockDim.y == gridDim.y - 1) &&
      j < jmax - 1)
    shared_P[local_idx + blockDim.x + 2] = P[global_idx + imax];

  __syncthreads();

  if (i < imax - 1 && j < jmax - 1 && (i + j) % 2 == color) {
    shared_P[local_idx] =
        (1.0 - omg) * shared_P[local_idx] +
        coeff * (Discretization::sor_helperSharedMem(shared_P, local_i, local_j,
                                                     blockDim.x + 2) -
                 RS[global_idx]);
    P[global_idx] = shared_P[local_idx];
  }
}

//__global__ void Residual_kernel_call(const double *P, const double *RS,
//                                     int imax, double jmax, double *residual)
//                                     {
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//  if (i > 0 && j > 0 && i < imax - 1 && j < jmax - 1) {
//    int idx = imax * j + i;
//    double val = Discretization::laplacian(P, i, j) - RS[idx];
//    atomicAdd(residual, (val * val));
//  }
//}

__global__ void residualKernelShared(const double *P, const double *RS,
                                           int imax, double jmax,
                                           double *residual_results) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int global_idx = j * imax + i;

  __shared__ double shared_P[(BLOCK_SIZE_RES + 2) * (BLOCK_SIZE_RES + 2)];
  __shared__ double shared_val[BLOCK_SIZE_RES * BLOCK_SIZE_RES];
  int shared_index = threadIdx.x * blockDim.y + threadIdx.y;
  shared_val[shared_index] = 0.0;

  int local_i = threadIdx.x + 1;
  int local_j = threadIdx.y + 1;
  int local_idx = local_j * (blockDim.x + 2) + local_i;

  // load the central part into shared memory
  if (local_i > 0 && local_j > 0 && local_i < blockDim.x + 1 &&
      local_j < blockDim.y + 1) {
    shared_P[local_idx] = P[global_idx];
  }

  // Left Halo
  if (threadIdx.x == 0 && i > 0) {
    shared_P[local_idx - 1] = P[global_idx - 1];
  }

  // Right Halo
  if ((threadIdx.x == blockDim.x - 1 || blockIdx.x == gridDim.x - 1) &&
      i < imax - 1) {
    shared_P[local_idx + 1] = P[global_idx + 1];
  }

  // Bottom Halo
  if (threadIdx.y == 0 && j > 0) {
    shared_P[local_idx - blockDim.x - 2] = P[global_idx - imax];
  }

  // Top Halo
  if ((threadIdx.y == blockDim.y - 1 || blockIdx.y == gridDim.y - 1) &&
      j < jmax - 1) {
    shared_P[local_idx + blockDim.x + 2] = P[global_idx + imax];
  }

  __syncthreads();

  if (i < imax - 1 && j < jmax - 1) {
    double val = Discretization::laplacianSharedMem(shared_P, local_i, local_j,
                                                    blockDim.x + 2) -
                 RS[global_idx];
    shared_val[shared_index] = val * val;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s /= 2) {
      if (shared_index < s) {
        shared_val[shared_index] =
            shared_val[shared_index] + shared_val[shared_index + s];
      }
      __syncthreads(); // Synchronize after each reduction step
    }
    // atomicAdd(residual_results, (val * val));
  }

  // Write the result of this block's max to global memory
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    residual_results[blockIdx.x * gridDim.y + blockIdx.y] = shared_val[0];
  }
}

double calculatePressureKernel(Matrix &P, const Matrix &RS, const Domain &domain,
                             double omg, double *d_rlocBlock, std::vector<double>& h_rlocBlock) {
  dim3 threadsPerBlock(BLOCK_SIZE_SOR, BLOCK_SIZE_SOR);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (domain.jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  size_t shared_mem_sor =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 1 * sizeof(double);

  double coeff =
      omg /
      (2.0 * (1.0 / (domain.dx * domain.dx) + 1.0 / (domain.dy * domain.dy)));

  SORKernelShared<<<numBlocks, threadsPerBlock, shared_mem_sor>>>(
      thrust::raw_pointer_cast(P.d_container.data()),
      thrust::raw_pointer_cast(RS.d_container.data()), domain.imax + 2,
      domain.jmax + 2, omg, coeff, 0);
  CHECK(cudaGetLastError());

  SORKernelShared<<<numBlocks, threadsPerBlock, shared_mem_sor>>>(
      thrust::raw_pointer_cast(P.d_container.data()),
      thrust::raw_pointer_cast(RS.d_container.data()), domain.imax + 2,
      domain.jmax + 2, omg, coeff, 1);
  CHECK(cudaGetLastError());
 
  double res = 0.0;
  //double h_rlocBlock[numBlocks.x * numBlocks.y];
  //std::vector<double> h_rlocBlock(numBlocks.x*numBlocks.y);
  double h_rloc = 0.0;

  size_t shared_mem_residual =
      (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) * 1 * sizeof(double) +
      threadsPerBlock.x * threadsPerBlock.y;

  residualKernelShared<<<numBlocks, threadsPerBlock,
                               shared_mem_residual>>>(
      thrust::raw_pointer_cast(P.d_container.data()),
      thrust::raw_pointer_cast(RS.d_container.data()), domain.imax + 2,
      domain.jmax + 2, d_rlocBlock);

  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(h_rlocBlock.data(), d_rlocBlock,
                   numBlocks.x * numBlocks.y * sizeof(double),
                   cudaMemcpyDeviceToHost));

  // Find the maximum in the result array
  for (int i = 0; i < numBlocks.x * numBlocks.y; ++i) {
    h_rloc = h_rloc + h_rlocBlock[i];
  }

  res = h_rloc / (domain.imax * domain.jmax);
  res = std::sqrt(res);

  return res;
}
} // namespace PressureSolverKernels
