#pragma once
#include "domain.hpp"
#include <cmath>

class DiscretizationHost {
public:
  DiscretizationHost(int imax, int jmax, double dx, double dy, double gamma);

  // Calculating the value of convective part of U
  static double convection_u(const double *U, const double *V,
                              int i, int j);

  // Calculating the value of convective part of V
  static double convection_v(const double *U, const double *V,
                              int i, int j);
  
  // Calculating the value of convective part of T
  static double convection_T(const double *U, const double *V,
                             const double *T, 
                             int i, int j);
  
  // Using the same for calculating diffusive part of U and V
  static double diffusion(const double *A,  int i,
                          int j);
  
  // Calculating the laplacian part of the equation

  static double laplacian(const double *P,  int i,
                          int j);
  
  // Calculating the SOR Helper
  static double sor_helper(const double *P,  int i,
                           int j);
  
  static double interpolate(const double *A, int i, int j, int i_offset,
                            int j_offset);

 static int _imax;
 static int _jmax;
 static double _dx;
 static double _dy;
 static double _gamma;
};