#pragma once

#include "domain.hpp"
#include "fields.hpp"
#include "boundary_kernels.hpp"

class Boundary {
public:
  Boundary(Fields *fields, Domain *domain, double Th, double Tc);
  ~Boundary();
  void applyBoundaries();

  void applyPressure();

  Fields *_fields;
  Domain *_domain;
  double _Th;
  double _Tc;
  cudaStream_t streamLR;
  cudaStream_t streamTB;
  cudaEvent_t eventLR;
  cudaEvent_t eventTB;
};


