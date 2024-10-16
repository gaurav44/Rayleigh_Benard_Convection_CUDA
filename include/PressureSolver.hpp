#pragma once

#include "DataStructure.hpp"
#include "Discretization.hpp"
#include "Domain.hpp"
#include "cuda_utils.hpp"

class PressureSolver {
public:
  PressureSolver(Domain *domain);

  ~PressureSolver();

  double calculate_pressure(Matrix &P, const Matrix &RS, const Domain &domain);
  static constexpr double eps = 0.00001; // convergence tolerance for SOR
  static const int itermax = 500;        // maximum iterations for SOR
  static constexpr double omg = 1.7;     // relaxation factor for SOR

  double *d_rloc;
  Domain *_domain;
};
extern double PressureSolver_kernel(Matrix &P, const Matrix &RS,
                                    const Domain &domain, double omg,
                                    double *d_rloc);
