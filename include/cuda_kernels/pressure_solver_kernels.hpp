#pragma once
#include "datastructure.hpp"
#include "domain.hpp"
#include "fields.hpp"
#include <vector>

namespace PressureSolverKernels {
extern double calculatePressureKernel(Matrix &P, const Matrix &RS,
                                    const Domain &domain, double omg,
                                    double *d_rlocBlock, std::vector<double>& h_rlocBlock);//, double* h_rlocBlock);
} // namespace PressureSolverKernels
