#include "Boundary.hpp"
#include "cuda_utils.hpp"

__global__ void BoundaryLR_kernel_call(double *U, double *V, double *F,
                                       double *G, double *T, int imax,
                                       int jmax) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < jmax) {
    int idxLeft0 = imax * j + 0;
    int idxLeft1 = imax * j + 1;

    // Left BC
    U[idxLeft0] = 0.0;
    V[idxLeft0] = -V[idxLeft1];
    F[idxLeft0] = U[idxLeft0];
    T[idxLeft0] = T[idxLeft1];

    int idxRight1 = imax * j + imax - 1;
    int idxRight2 = imax * j + imax - 2;

    // Right BC
    U[idxRight2] = 0.0;
    V[idxRight1] = -V[idxRight2];
    F[idxRight2] = U[idxRight2];
    T[idxRight1] = T[idxRight2];
  }
}

__global__ void BoundaryTB_kernel_call(double *U, double *V, double *F,
                                       double *G, double *T, int imax, int jmax,
                                       double Th, double Tc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < imax) {
    int idxBottom0 = imax * 0 + i;
    int idxBottom1 = imax * 1 + i;

    // Bottom BC
    V[idxBottom0] = 0;
    U[idxBottom0] = -U[idxBottom1];
    G[idxBottom0] = V[idxBottom0];
    T[idxBottom0] = 2 * Th - T[idxBottom1];

    int idxTop1 = imax * (jmax - 1) + i;
    int idxTop2 = imax * (jmax - 2) + i;

    // Top BC
    V[idxTop2] = 0;
    U[idxTop1] = U[idxTop2];
    G[idxTop2] = V[idxTop2];
    T[idxTop1] = 2 * Tc - T[idxTop2];
  }
}

void Boundary_kernel(Fields &fields, const Domain &domain, double Th, double Tc,
                     cudaStream_t streamLR,cudaStream_t streamTB, cudaEvent_t eventLR,
                     cudaEvent_t eventTB) {
  dim3 threadsPerBlock(1024);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x);

  //cudaStream_t streamLR;
  //cudaStream_t streamTB;
  //CHECK(cudaStreamCreate(&streamLR));
  //CHECK(cudaStreamCreate(&streamTB));

  //cudaEvent_t eventLR, eventTB;
  //CHECK(cudaEventCreate(&eventLR));
  //CHECK(cudaEventCreate(&eventTB));

  BoundaryLR_kernel_call<<<numBlocks, threadsPerBlock, 0, streamLR>>>(
      thrust::raw_pointer_cast(fields.U.d_container.data()),
      thrust::raw_pointer_cast(fields.V.d_container.data()),
      thrust::raw_pointer_cast(fields.F.d_container.data()),
      thrust::raw_pointer_cast(fields.G.d_container.data()),
      thrust::raw_pointer_cast(fields.T.d_container.data()), domain.imax + 2,
      domain.jmax + 2);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(eventLR, streamLR));

  BoundaryTB_kernel_call<<<numBlocks, threadsPerBlock, 0, streamTB>>>(
      thrust::raw_pointer_cast(fields.U.d_container.data()),
      thrust::raw_pointer_cast(fields.V.d_container.data()),
      thrust::raw_pointer_cast(fields.F.d_container.data()),
      thrust::raw_pointer_cast(fields.G.d_container.data()),
      thrust::raw_pointer_cast(fields.T.d_container.data()), domain.imax + 2,
      domain.jmax + 2, Th, Tc);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(eventTB, streamTB));

  CHECK(cudaStreamWaitEvent(0, eventLR, 0));
  CHECK(cudaStreamWaitEvent(0, eventTB, 0));

  //CHECK(cudaStreamDestroy(streamTB));
  //CHECK(cudaStreamDestroy(streamLR));
  //CHECK(cudaEventDestroy(eventTB));
  //CHECK(cudaEventDestroy(eventLR));
  // cudaDeviceSynchronize();
}

__global__ void BoundaryP_kernel_call(double *P, int imax, int jmax) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j > 0 && j < jmax) {
    int idxLeft0 = imax * j + 0;
    int idxLeft1 = imax * j + 1;
    // Left BC
    P[idxLeft0] = P[idxLeft1];

    int idxRight1 = imax * j + imax - 1;
    int idxRight2 = imax * j + imax - 2;
    // Right BC
    P[idxRight1] = P[idxRight2];
  }

  int i = j;
  if (i > 0 && i < imax) {
    int idxBottom0 = imax * 0 + i;
    int idxBottom1 = imax * 1 + i;
    // Bottom BC
    P[idxBottom0] = P[idxBottom1];

    int idxTop1 = imax * (jmax - 1) + i;
    int idxTop2 = imax * (jmax - 2) + i;
    // Top BC
    P[idxTop1] = P[idxTop2];
  }
}

void BoundaryP_kernel(Matrix &p, const Domain &domain) {
  dim3 threadsPerBlock(256);
  dim3 numBlocks((domain.imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x);

  BoundaryP_kernel_call<<<numBlocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(p.d_container.data()), domain.imax + 2,
      domain.jmax + 2);
  CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();
}
