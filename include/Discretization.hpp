#pragma once
#include "DataStructure.hpp"
#include "Domain.hpp"
#include "thrust/device_vector.h"
#include <cmath>
class Discretization {
public:
  // Calculating the value of convective part of U
  static double convection_u(const Matrix &U, const Matrix &V,
                             const Domain &domain, int i, int j);

  __device__ static double convection_u(const double *U, const double *V,
                                        double dx, double dy, int i, int j,
                                        double gamma, int imax);
  // Calculating the value of convective part of V
  static double convection_v(const Matrix &U, const Matrix &V,
                             const Domain &domain, int i, int j);

  __device__ static double convection_v(const double *U, const double *V,
                                        double dx, double dy, int i, int j,
                                        double gamma, int imax);

  // Calculating the value of convective part of T
  static double convection_T(const Matrix &U, const Matrix &V, const Matrix &T,
                             const Domain &domain, int i, int j);

  __device__ static double convection_T(const double *U, const double *V,
                                        const double *T, double gamma,
                                        double dx, double dy, int i, int j,
                                        int imax);

  // Using the same for calculating diffusive part of U and V
  static double diffusion(const Matrix &A, const Domain &domain, int i, int j);
  __device__ static double diffusion(const double *A, double dx, double dy,
                                     int i, int j, int imax);

  // Calculating the laplacian part of the equation
  static double laplacian(const Matrix &P, const Domain &domain, int i, int j);
  __device__ static double laplacian(const double* P, double dx, double dy, int i, int j, int imax);

  // Calculating the SOR Helper
  static double sor_helper(const Matrix &P, const Domain &domain, int i, int j);

  __device__ static double sor_helper(const double* P, double dx, double dy, int i, int j, int imax);

  static double interpolate(const Matrix &A, int i, int j, int i_offset,
                            int j_offset);

  __device__ static double interpolate(const double *A, int i, int j,
                                       int i_offset, int j_offset, int imax);
};
