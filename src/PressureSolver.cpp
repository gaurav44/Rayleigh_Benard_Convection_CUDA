#include "PressureSolver.hpp"

PressureSolver::PressureSolver() { CHECK(cudaMalloc(&d_rloc, sizeof(double))); }

PressureSolver::~PressureSolver() { CHECK(cudaFree(d_rloc)); }

double PressureSolver::calculate_pressure(Matrix &P, const Matrix &RS,
                                          const Domain &domain) {
  double res = PressureSolver_kernel(P, RS, domain, omg, d_rloc);

  return res;
};
