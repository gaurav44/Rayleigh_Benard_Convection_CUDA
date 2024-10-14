#pragma once

#include "DataStructure.hpp"
#include "Discretization.hpp"
#include "Domain.hpp"

class PressureSolver {
public:
  static double calculate_pressure(Matrix &P, const Matrix &RS,
                                   const Domain &domain, double omg);
};
