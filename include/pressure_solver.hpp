#pragma once

#include "datastructure.hpp"
#include "discretization.hpp"
#include "domain.hpp"
#include "block_sizes.hpp"
#include "cuda_utils.hpp"
#include "pressure_solver_kernels.hpp"

class PressureSolver {
public:
  PressureSolver(Domain *domain);

  ~PressureSolver();

  double calculatePressure(Matrix &P, const Matrix &RS, const Domain &domain);
  static constexpr double eps = 0.00001; // convergence tolerance for SOR
  static const int itermax = 500;        // maximum iterations for SOR
  static constexpr double omg = 1.7;     // relaxation factor for SOR

  double *d_rlocBlock;
  std::vector<double> h_rlocBlock;
  Domain *_domain;
};
