#include "Boundary.hpp"

Boundary::Boundary(Fields *fields, Domain *domain, double Th, double Tc)
    : _fields(fields), _domain(domain), _Th(Th), _Tc(Tc) {}

void Boundary::apply_boundaries() {
  Boundary_kernel(*_fields, *_domain, _Th, _Tc);
}

void Boundary::apply_pressure() { BoundaryP_kernel(_fields->P, *_domain); }
