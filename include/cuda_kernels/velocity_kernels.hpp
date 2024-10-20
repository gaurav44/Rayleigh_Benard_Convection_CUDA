#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace VelocityKernels {
extern void calculateVelocitiesKernel(Matrix &U, Matrix &V, const Matrix &F, const Matrix &G,
                      const Matrix &P, const Domain &domain);
} // namespace VelocityKernels
