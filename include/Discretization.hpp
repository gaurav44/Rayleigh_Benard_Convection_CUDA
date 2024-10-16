#pragma once
#include "DataStructure.hpp"
#include "Domain.hpp"
#include "thrust/device_vector.h"
#include <cmath>

class Discretization {

public:
  Discretization(int imax, int jmax, double dx, double dy, double gamma);

  __device__ static double convection_u(const double *U, const double *V, int i,
                                        int j);

  __device__ static double convection_uSharedMem(const double *U,
                                                 const double *V, int i, int j,
                                                 int imax);

  __device__ static double convection_v(const double *U, const double *V, int i,
                                        int j);

  __device__ static double convection_vSharedMem(const double *U, const double *V, int i,
                                        int j, int imax);

  __device__ static double convection_T(const double *U, const double *V,
                                        const double *T, int i, int j);
  __device__ static double convection_TSharedMem(const double *U,
                                                 const double *V,
                                                 const double *T, int i, int j,
                                                 int imax);

  // Using the same for calculating diffusive part of U and V
  __device__ static double diffusion(const double *A, int i, int j);
  __device__ static double diffusionSharedMem(const double *A, int i, int j,
                                              int imax);

  // Calculating the laplacian part of the equation
  __device__ static double laplacian(const double *P, int i, int j);
  __device__ static double laplacianSharedMem(const double *P, int i, int j, int imax);

  // Calculating the SOR Helper
  __device__ static double sor_helper(const double *P, int i, int j);
  __device__ static double sor_helperSharedMem(const double *P, int i, int j, int imax);

  __device__ static double interpolate(const double *A, int i, int j,
                                       int i_offset, int j_offset);

  __device__ static double interpolateSharedMem(const double *A, int i, int j,
                                       int i_offset, int j_offset, int imax);

};
// Maybe move this into a global struct of constants
static __constant__ int _imax;
static __constant__ int _jmax;
static __constant__ double _dx;
static __constant__ double _dy;
static __constant__ double _gamma;
