#pragma once
#include "datastructure.hpp"

class Fields {
public:
   Fields(int imax, int jmax, double T_inf) : U(Matrix(imax, jmax)),
                                              V(Matrix(imax, jmax)),
                                              F(Matrix(imax, jmax)),
                                              G(Matrix(imax, jmax)),
                                              P(Matrix(imax, jmax)),
                                              RS(Matrix(imax, jmax)),
                                              T(Matrix(imax, jmax, T_inf)) {}


  Matrix U;
  Matrix V;
  Matrix F;
  Matrix G;
  Matrix T;
  Matrix P;
  Matrix RS;
};
