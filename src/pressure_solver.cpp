#include "pressure_solver.hpp"

PressureSolver::PressureSolver(Domain* domain) : _domain(domain){}

double PressureSolver::calculatePressure(Matrix &P, const Matrix &RS) {
  double dx = _domain->dx;
  double dy = _domain->dy;
  double coeff =
      omg / (2.0 * (1.0 / (dx * dx) +
                    1.0 / (dy * dy))); // = _omega * h^2 / 4.0, if dx == dy == h

  for (int i = 1; i < _domain->imax + 1; i++) {
    for (int j = 1; j < _domain->jmax + 1; j++) {
      P(i, j) =
          (1.0 - omg) * P(i, j) +
          coeff * (Discretization::sor_helper(P, *_domain, i, j) - RS(i, j));
    }
  }

  double res = 0.0;
  double rloc = 0.0;

  // Using squared value of difference to calculate residual
  for (int i = 1; i < _domain->imax + 1; i++) {
    for (int j = 1; j < _domain->jmax + 1; j++) {
      double val = Discretization::laplacian(P, *_domain, i, j) - RS(i, j);
      rloc += (val * val);
    }
  }
  res = rloc / (_domain->imax * _domain->jmax);
  res = std::sqrt(res);

  return res;
};
