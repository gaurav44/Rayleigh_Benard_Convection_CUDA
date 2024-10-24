#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace TimestepKernels {
extern std::pair<double, double>
calculateUVMaxKernel(const Matrix &U, const Matrix &V, const Domain &domain,
          double *d_uBlockMax, double *d_vBlockMax, double *h_uBlockMax,
          double *h_vBlockMax);
__global__ void velocityUMaxKernel(const double *U, int imax, int jmax,
                                   double *max_results);
__global__ void velocityVMaxKernel(const double *V, int imax, int jmax,
                                   double *max_results);
} // namespace TimestepKernels
