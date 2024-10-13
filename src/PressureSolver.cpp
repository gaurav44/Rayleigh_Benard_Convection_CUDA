#include "PressureSolver.hpp"

double PressureSolver::calculate_pressure(Matrix &P, const Matrix &RS,
                                          const Domain &domain) {
  double res = PressureSolver_kernel(P, RS, domain, omg);

  return res;
};
