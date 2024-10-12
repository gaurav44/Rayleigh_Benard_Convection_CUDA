#include "Boundary.hpp"

void Boundary::apply_boundaries(Fields &fields, const Domain &domain, double Th,
                                double Tc) {
  fields.U.copyToDevice();
  fields.V.copyToDevice();
  fields.F.copyToDevice();
  fields.G.copyToDevice();
  fields.T.copyToDevice();
  Boundary_kernel(fields, domain, Th, Tc);
  fields.U.copyToHost();
  fields.V.copyToHost();
  fields.F.copyToHost();
  fields.G.copyToHost();
  fields.T.copyToHost();

  // int imaxb = domain.imax + 2;
  // int jmaxb = domain.jmax + 2;

  // for (int j = 0; j < jmaxb; j++) {
  //   // Left BC
  //   fields.U(0, j) = 0;
  //   fields.V(0, j) = -fields.V(1, j);
  //   fields.F(0, j) = fields.U(0, j);
  //   fields.T(0, j) = fields.T(1, j);

  //  // Right BC
  //  fields.U(imaxb - 2, j) = 0;
  //  fields.V(imaxb - 1, j) = -fields.V(imaxb - 2, j);
  //  fields.F(imaxb - 2, j) = fields.U(imaxb - 2, j);
  //  fields.T(imaxb - 1, j) = fields.T(imaxb - 2, j);
  //}

  // for (int i = 0; i < imaxb; i++) {
  //   // Bottom BC
  //   fields.V(i, 0) = 0;
  //   fields.U(i, 0) = -fields.U(i, 1);
  //   fields.G(i, 0) = fields.V(i, 0);
  //   fields.T(i, 0) = 2 * Th - fields.T(i, 1);

  //  // Top BC
  //  fields.V(i, jmaxb - 2) = 0;
  //  fields.U(i, jmaxb - 1) = fields.U(i, jmaxb - 2);
  //  fields.G(i, jmaxb - 2) = fields.V(i, jmaxb - 2);
  //  fields.T(i, jmaxb - 1) = 2 * Tc - fields.T(i, jmaxb - 2);
  //}
}

void Boundary::apply_pressure(Matrix &p, const Domain &domain) {
  p.copyToDevice();
  BoundaryP_kernel(p, domain);
  p.copyToHost();
  //int imaxb = domain.imax + 2;
  //int jmaxb = domain.jmax + 2;
  //for (int j = 0; j < jmaxb; j++) {
  //  // Left BC
  //  p(0, j) = p(1, j);

  //  // Right BC
  //  p(imaxb - 1, j) = p(imaxb - 2, j);
  //}

  //for (int i = 0; i < imaxb; i++) {
  //  // Bottom BC
  //  p(i, 0) = p(i, 1);

  //  // Top BC
  //  p(i, jmaxb - 1) = p(i, jmaxb - 2);
  //}
}
