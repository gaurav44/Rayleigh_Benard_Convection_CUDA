#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace VelocityKernels {
extern void calculateVelocitiesKernel(Matrix &U, Matrix &V, const Matrix &F, const Matrix &G,
                      const Matrix &P, const Domain &domain);
__global__ void U_kernelShared_call(double *U, const double *F, const double *P,
                                    double dx, int imax, double jmax,
                                    double dt);
__global__ void V_kernelShared_call(double *V, const double *G, const double *P,
                                    double dy, int imax, int jmax,
                                    double dt);
__global__ void velocityKernelShared(double *U, double *V, const double *F,
                                     const double *G, const double *P,
                                     double dx, double dy, int imax,
                                     double jmax, double dt);
} // namespace VelocityKernels
