#pragma once

#include "datastructure.hpp"
#include "discretization.hpp"
#include "domain.hpp"

class PressureSolver {
public:
  PressureSolver(Domain *domain);

  double calculatePressure(Matrix &P, const Matrix &RS);

  static constexpr double eps = 0.00001; // convergence tolerance for SOR
  static const int itermax = 500;        // maximum iterations for SOR
  static constexpr double omg = 1.7;     // relaxation factor for SOR

  Domain* _domain;
};
