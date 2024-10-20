#pragma once
#include "datastructure.hpp"
#include "domain.hpp"

namespace TemperatureKernels
{
    extern void calculateTemperatureKernel(const Matrix &U, const Matrix &V, Matrix &T,
                               const Domain &domain);
} // namespace TemperatureKernels
