#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace TemperatureKernels
{
    extern void calculateTemperatureKernel(const Matrix &U, const Matrix &V, Matrix &T,
                               const Domain &domain);
    __global__ void temperatureKernelShared(const double *U, const double *V,
                                        double *T, int imax, int jmax,
                                        double alpha, double dt);
    __global__ void temperature_kernel_call(const double *U, const double *V,
                                       double *T, const double *T_old,
                                       double dx, double dy, int imax,
                                       int jmax, double gamma, double
                                       alpha, double dt);
} // namespace TemperatureKernels
