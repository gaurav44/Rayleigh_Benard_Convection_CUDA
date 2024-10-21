#include "boundary.hpp"

Boundary::Boundary(Fields* fields, Domain* domain, double Th, double Tc) : _fields(fields), _domain(domain), _Th(Th), _Tc(Tc){}

void Boundary::applyBoundaries(){

    int imaxb = _domain->imax + 2;
    int jmaxb = _domain->jmax + 2;

    for (int j = 0; j < jmaxb; j++) {
      // Left BC
      _fields->U(0, j) = 0;
      _fields->V(0, j) = -_fields->V(1, j);
      _fields->F(0, j) = _fields->U(0, j);
      _fields->T(0, j) = _fields->T(1, j);

      // Right BC
      _fields->U(imaxb - 2, j) = 0;
      _fields->V(imaxb - 1, j) = -_fields->V(imaxb - 2, j);
      _fields->F(imaxb - 2, j) = _fields->U(imaxb - 2, j);
      _fields->T(imaxb - 1, j) = _fields->T(imaxb - 2, j);
    }

    for (int i = 0; i < imaxb; i++) {
      // Bottom BC
      _fields->V(i, 0) = 0;
      _fields->U(i, 0) = -_fields->U(i, 1);
      _fields->G(i, 0) = _fields->V(i, 0);
      _fields->T(i, 0) = 2 * _Th - _fields->T(i, 1);

      // Top BC
      _fields->V(i, jmaxb - 2) = 0;
      _fields->U(i, jmaxb - 1) = _fields->U(i, jmaxb - 2);
      _fields->G(i, jmaxb - 2) = _fields->V(i, jmaxb - 2);
      _fields->T(i, jmaxb - 1) = 2 * _Tc - _fields->T(i, jmaxb - 2);
    }
}

void Boundary::applyPressureBoundary() {
    int imaxb = _domain->imax + 2;
    int jmaxb = _domain->jmax + 2;
    for (int j = 0; j < jmaxb; j++) {
      // Left BC
      _fields->P(0, j) = _fields->P(1, j);

      // Right BC
      _fields->P(imaxb - 1, j) = _fields->P(imaxb - 2, j);
    }

    for (int i = 0; i < imaxb; i++) {
      // Bottom BC
      _fields->P(i, 0) = _fields->P(i, 1);

      // Top BC
      _fields->P(i, jmaxb - 1) = _fields->P(i, jmaxb - 2);
    }
}