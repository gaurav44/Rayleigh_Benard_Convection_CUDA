#pragma once
#include "DataStructure.hpp"
#include "Domain.hpp"
#include "thrust/device_vector.h"
#include <cmath>

class Discretization {

public:
  Discretization(int imax, int jmax, double dx, double dy, double gamma);

  // Calculating the value of convective part of U
  static double convection_u(const Matrix &U, const Matrix &V,
                             const Domain &domain, int i, int j);

  __device__ static double convection_u(const double *U, const double *V, int i,
                                        int j);
  // Calculating the value of convective part of V
  static double convection_v(const Matrix &U, const Matrix &V,
                             const Domain &domain, int i, int j);

  __device__ static double convection_v(const double *U, const double *V, int i,
                                        int j);

  // Calculating the value of convective part of T
  static double convection_T(const Matrix &U, const Matrix &V, const Matrix &T,
                             const Domain &domain, int i, int j);

  __device__ static double convection_T(const double *U, const double *V,
                                        const double *T, int i, int j);

  // Using the same for calculating diffusive part of U and V
  static double diffusion(const Matrix &A, const Domain &domain, int i, int j);
  __device__ static double diffusion(const double *A, int i, int j);

  // Calculating the laplacian part of the equation
  static double laplacian(const Matrix &P, const Domain &domain, int i, int j);
  __device__ static double laplacian(const double *P, int i, int j);

  // Calculating the SOR Helper
  static double sor_helper(const Matrix &P, const Domain &domain, int i, int j);

  __device__ static double sor_helper(const double *P, int i, int j);

  static double interpolate(const Matrix &A, int i, int j, int i_offset,
                            int j_offset);

  __device__ static double interpolate(const double *A, int i, int j,
                                       int i_offset, int j_offset);

};
//Maybe move this into a global struct of constants
static __constant__ int _imax;
static __constant__ int _jmax;
static __constant__ double _dx;
static __constant__ double _dy;
static __constant__ double _gamma;
