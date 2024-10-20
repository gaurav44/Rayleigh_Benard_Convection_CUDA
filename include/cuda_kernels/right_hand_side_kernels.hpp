#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace RightHandSideKernels {

extern void calculateRightHandSideKernel(const Matrix &F, const Matrix &G, Matrix &RS,
                      const Domain &domain);
} // namespace RightHandSideKernels
