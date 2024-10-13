#pragma once

#include "Domain.hpp"
#include "Fields.hpp"

class Boundary {
public:
  Boundary(Fields* fields);
  void apply_boundaries(const Domain &domain, double Th,
                               double Tc);

  void apply_pressure(const Domain &domain);

  Fields* _fields;
};

extern void Boundary_kernel(Fields &fields, const Domain &domain, double Th,
                            double Tc);
extern void BoundaryP_kernel(Matrix &p, const Domain &domain);
