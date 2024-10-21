#pragma once

#include "domain.hpp"
#include "fields.hpp"

class Boundary {
public:
  Boundary(Fields* fields, Domain* domain, double Th, double Tc);
  void applyBoundaries();

  void applyPressureBoundary();

  Fields *_fields;
  Domain *_domain;
  double _Th;
  double _Tc;
};
