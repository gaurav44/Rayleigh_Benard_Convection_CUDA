#pragma once
#include "datastructure.hpp"
#include "domain.hpp"
#include "fields.hpp"
#include <vector>

namespace PressureSolverKernels {
extern double calculatePressureKernel(Matrix &P, const Matrix &RS,
                                    const Domain *domain, double omg,
                                    double *d_rlocBlock, std::vector<double>& h_rlocBlock);//, double* h_rlocBlock);
__global__ void SORKernelShared(double *P, const double *RS, int imax, int jmax,
                                double omg, double coeff, int color);
__global__ void residualKernelShared(const double *P, const double *RS,
                                     int imax, int jmax,
                                     double *residual_results);
} // namespace PressureSolverKernels
