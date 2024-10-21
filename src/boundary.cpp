#include "boundary.hpp"
#include "cuda_utils.hpp"

Boundary::Boundary(Fields *fields, Domain *domain, double Th, double Tc)
    : _fields(fields), _domain(domain), _Th(Th), _Tc(Tc) {
  CHECK(cudaStreamCreate(&streamLR));
  CHECK(cudaStreamCreate(&streamTB));
  CHECK(cudaEventCreate(&eventLR));
  CHECK(cudaEventCreate(&eventTB));
}

Boundary::~Boundary() {
  CHECK(cudaStreamDestroy(streamTB));
  CHECK(cudaStreamDestroy(streamLR));
  CHECK(cudaEventDestroy(eventTB));
  CHECK(cudaEventDestroy(eventLR));
}

void Boundary::applyBoundaries() {
  BoundaryKernels::applyBoundaryKernel(*_fields, *_domain, _Th, _Tc, streamLR, streamTB, eventLR, eventTB);
}

void Boundary::applyPressure() { BoundaryKernels::applyPressureBoundaryKernel(_fields->P, *_domain); }
