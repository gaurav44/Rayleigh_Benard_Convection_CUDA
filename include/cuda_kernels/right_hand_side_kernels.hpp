#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace RightHandSideKernels {
extern void calculateRightHandSideKernel(const Matrix &F, const Matrix &G, Matrix &RS,
                      const Domain &domain);
__global__ void rightHandSideKernelShared(const double *F, const double *G,
                                          double *RS, double dx, double dy, int imax,
                                          int jmax, double dt);
__global__ void RS_kernel_call(const double *F, const double *G, double *RS,
                              double dx, double dy, int imax, double jmax,
                              double dt);
} // namespace RightHandSideKernels
