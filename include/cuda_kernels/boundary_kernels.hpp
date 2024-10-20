#pragma once
#include "datastructure.hpp"
#include "domain.hpp"
#include "fields.hpp"

namespace BoundaryKernels {
extern void applyBoundaryKernel(Fields &fields, const Domain &domain, double Th,
                            double Tc, cudaStream_t streamLR,
                            cudaStream_t streamTB, cudaEvent_t eventLR,
                            cudaEvent_t eventTB);
extern void applyPressureBoundaryKernel(Matrix &p, const Domain &domain);
} // namespace BoundaryKernels
