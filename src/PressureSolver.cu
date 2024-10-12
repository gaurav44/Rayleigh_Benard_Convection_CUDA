#include "PressureSolver.hpp"

double PressureSolver::calculate_pressure(Matrix &P,
                                          const Matrix &RS,
                                          const Domain &domain, double omg) {
  //double res = PressureSolver_kernel(P, RS, domain, omg);
  double dx = domain.dx;
  double dy = domain.dy;
  double coeff =
      omg / (2.0 * (1.0 / (dx * dx) +
                    1.0 / (dy * dy))); // = _omega * h^2 / 4.0, if dx == dy == h

  for (int i = 1; i < domain.imax + 1; i++) {
    for (int j = 1; j < domain.jmax + 1; j++) {
      P(i, j) =
          (1.0 - omg) * P(i, j) +
          coeff * (Discretization::sor_helper(P, domain, i, j) - RS(i, j));
    }
  }
  // std::cout << "\n";
  // P.printField();

  double res = 0.0;
  double rloc = 0.0;

  // Using squared value of difference to calculate residual
  for (int i = 1; i < domain.imax + 1; i++) {
    for (int j = 1; j < domain.jmax + 1; j++) {
      double val = Discretization::laplacian(P, domain, i, j) - RS(i, j);
      rloc += (val * val);
    }
  }
  res = rloc / (domain.imax * domain.jmax);
  res = std::sqrt(res);

  return res;
};
