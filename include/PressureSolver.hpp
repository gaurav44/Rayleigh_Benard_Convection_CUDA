#pragma once

#include "Domain.hpp"
#include "DataStructure.hpp"
#include "Discretization.hpp"

class PressureSolver {
    public:
        static double calculate_pressure(Matrix& P,
                                         const Matrix& RS,
                                         const Domain& domain,
                                         double omg );
};
extern double PressureSolver_kernel(Matrix &P, const Matrix &RS, const Domain &domain, double omg);
