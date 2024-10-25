#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace FluxesKernels
{
   extern void calculateFluxesKernel(const Matrix &U, const Matrix &V, Matrix &F, Matrix &G,
                        const Matrix &T, const Domain &domain);

   __global__ void FluxesKernelShared(const double *U, const double *V,
                                   const double *T, double *F, double *G,
                                   int imax, int jmax, double nu, double dt,
                                   double GX, double GY, double beta);
} // namespace FluxesKernels
