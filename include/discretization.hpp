#pragma once
#include "datastructure.hpp"
#include "domain.hpp"
#include <cmath>

class Discretization {
public:
  Discretization(int imax, int jmax, double dx, double dy, double gamma);

  // Calculating the value of convective part of U
  static double convection_u(const Matrix &U, const Matrix &V,
                             const Domain &domain, int i, int j);

  // Calculating the value of convective part of V
  static double convection_v(const Matrix&U, const Matrix&V,
                             const Domain &domain, int i, int j);
  
  // Calculating the value of convective part of T
  static double convection_T(const Matrix&U, const Matrix&V,
                             const Matrix&T, const Domain &domain,
                             int i, int j);
  
  // Using the same for calculating diffusive part of U and V
  static double diffusion(const Matrix &A, const Domain &domain, int i,
                          int j);
  
  // Calculating the laplacian part of the equation

  static double laplacian(const Matrix &P, const Domain &domain, int i,
                          int j);
  
  // Calculating the SOR Helper
  static double sor_helper(const Matrix &P, const Domain &domain, int i,
                           int j);
  
  static double interpolate(const Matrix &A, int i, int j, int i_offset,
                            int j_offset);

 static int _imax;
 static int _jmax;
 static double _dx;
 static double _dy;
 static double _gamma;
};
