#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace TimestepKernels {
extern std::pair<double, double>
calculateTimeStepKernel(const Matrix &U, const Matrix &V, const Domain &domain,
          double *d_uBlockMax, double *d_vBlockMax, double *h_uBlockMax,
          double *h_vBlockMax);
} // namespace TimestepKernels
