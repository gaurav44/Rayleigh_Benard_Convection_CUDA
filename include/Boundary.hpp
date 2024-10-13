#pragma once

#include "Domain.hpp"
#include "Fields.hpp"

class Boundary {
public:
  Boundary(Fields *fields, Domain *domain, double Th, double Tc);
  void apply_boundaries();

  void apply_pressure();

  Fields *_fields;
  Domain *_domain;
  double _Th;
  double _Tc;
};

extern void Boundary_kernel(Fields &fields, const Domain &domain, double Th,
                            double Tc);
extern void BoundaryP_kernel(Matrix &p, const Domain &domain);
