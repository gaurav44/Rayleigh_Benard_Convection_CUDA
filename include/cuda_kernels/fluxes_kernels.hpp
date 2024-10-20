#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace FluxesKernels
{
   extern void calculateFluxesKernel(const Matrix &U, const Matrix &V, Matrix &F, Matrix &G,
                        const Matrix &T, const Domain &domain);
} // namespace FluxesKernels
