#pragma once

#include "DataStructure.hpp"
#include "Discretization.hpp"
#include "Domain.hpp"

class PressureSolver {
public:
  static double calculate_pressure(Matrix &P, const Matrix &RS,
                                   const Domain &domain);
  static constexpr double eps = 0.00001; // convergence tolerance for SOR
  static const int itermax = 500;        // maximum iterations for SOR
  static constexpr double omg = 1.7;     // relaxation factor for SOR
};
extern double PressureSolver_kernel(Matrix &P, const Matrix &RS,
                                    const Domain &domain, double omg);
