#include "Discretization.hpp"

Discretization::Discretization(int imax, int jmax, double dx, double dy,
                               double gamma) {
  cudaMemcpyToSymbol(_imax, &imax, sizeof(int));
  cudaMemcpyToSymbol(_jmax, &jmax, sizeof(int));
  cudaMemcpyToSymbol(_dx, &dx, sizeof(double));
  cudaMemcpyToSymbol(_dy, &dy, sizeof(double));
  cudaMemcpyToSymbol(_gamma, &gamma, sizeof(double));
}

__device__ double Discretization::convection_u(const double *U, const double *V,
                                               int i, int j) {

  int idx = _imax * j + i;
  int idx_right = _imax * j + (i + 1);
  int idx_left = _imax * j + (i - 1);

  double term1 =
      (1 / _dx) * (interpolate(U, i, j, 1, 0) * interpolate(U, i, j, 1, 0) -
                   interpolate(U, i, j, -1, 0) * interpolate(U, i, j, -1, 0)) +
      (_gamma / _dx) *
          (fabs(interpolate(U, i, j, 1, 0)) * (U[idx] - U[idx_right]) / 2 -
           fabs(interpolate(U, i, j, -1, 0)) * (U[idx_left] - U[idx]) / 2);

  int idx_top = _imax * (j + 1) + i;
  int idx_bottom = _imax * (j - 1) + i;
  double term2 =
      (1 / _dy) *
          (interpolate(V, i, j, 1, 0) * interpolate(U, i, j, 0, 1) -
           interpolate(V, i, j - 1, 1, 0) * interpolate(U, i, j, 0, -1)) +
      (_gamma / _dy) *
          (fabs(interpolate(V, i, j, 1, 0)) * (U[idx] - U[idx_top]) / 2 -
           fabs(interpolate(V, i, j - 1, 1, 0)) * (U[idx_bottom] - U[idx]) / 2);
  return term1 + term2;
}

__device__ double Discretization::convection_uSharedMem(const double *U,
                                                        const double *V, int i,
                                                        int j, int imax) {

  int idx = imax * j + i;
  int idx_right = imax * j + (i + 1);
  int idx_left = imax * j + (i - 1);

  double term1 =
      (1 / _dx) * (interpolateSharedMem(U, i, j, 1, 0, imax) *
                       interpolateSharedMem(U, i, j, 1, 0, imax) -
                   interpolateSharedMem(U, i, j, -1, 0, imax) *
                       interpolateSharedMem(U, i, j, -1, 0, imax)) +
      (_gamma / _dx) * (fabs(interpolateSharedMem(U, i, j, 1, 0, imax)) *
                            (U[idx] - U[idx_right]) / 2 -
                        fabs(interpolateSharedMem(U, i, j, -1, 0, imax)) *
                            (U[idx_left] - U[idx]) / 2);

  int idx_top = imax * (j + 1) + i;
  int idx_bottom = imax * (j - 1) + i;
  double term2 =
      (1 / _dy) * (interpolateSharedMem(V, i, j, 1, 0, imax) *
                       interpolateSharedMem(U, i, j, 0, 1, imax) -
                   interpolateSharedMem(V, i, j - 1, 1, 0, imax) *
                       interpolateSharedMem(U, i, j, 0, -1, imax)) +
      (_gamma / _dy) * (fabs(interpolateSharedMem(V, i, j, 1, 0, imax)) *
                            (U[idx] - U[idx_top]) / 2 -
                        fabs(interpolateSharedMem(V, i, j - 1, 1, 0, imax)) *
                            (U[idx_bottom] - U[idx]) / 2);
  return term1 + term2;
}

__device__ double Discretization::convection_v(const double *U, const double *V,
                                               int i, int j) {
  int idx = _imax * j + i;
  int idx_right = _imax * j + (i + 1);
  int idx_left = _imax * j + (i - 1);
  int idx_top = _imax * (j + 1) + i;
  int idx_bottom = _imax * (j - 1) + i;

  double term1 =
      (1 / _dy) * (interpolate(V, i, j, 0, 1) * interpolate(V, i, j, 0, 1) -
                   interpolate(V, i, j, 0, -1) * interpolate(V, i, j, 0, -1)) +
      (_gamma / _dy) *
          (fabs(interpolate(V, i, j, 0, 1)) * (V[idx] - V[idx_top]) / 2 -
           fabs(interpolate(V, i, j, 0, -1)) * (V[idx_bottom] - V[idx]) / 2);

  double term2 =
      (1 / _dx) *
          (interpolate(U, i, j, 0, 1) * interpolate(V, i, j, 1, 0) -
           interpolate(U, i - 1, j, 0, 1) * interpolate(V, i, j, -1, 0)) +
      (_gamma / _dx) *
          (fabs(interpolate(U, i, j, 0, 1)) * (V[idx] - V[idx_right]) / 2 -
           fabs(interpolate(U, i - 1, j, 0, 1)) * (V[idx_left] - V[idx]) / 2);

  return term1 + term2;
}

__device__ double Discretization::convection_vSharedMem(const double *U,
                                                        const double *V, int i,
                                                        int j, int imax) {
  int idx = imax * j + i;
  int idx_right = imax * j + (i + 1);
  int idx_left = imax * j + (i - 1);
  int idx_top = imax * (j + 1) + i;
  int idx_bottom = imax * (j - 1) + i;

  double term1 =
      (1 / _dy) * (interpolateSharedMem(V, i, j, 0, 1, imax) *
                       interpolateSharedMem(V, i, j, 0, 1, imax) -
                   interpolateSharedMem(V, i, j, 0, -1, imax) *
                       interpolateSharedMem(V, i, j, 0, -1, imax)) +
      (_gamma / _dy) * (fabs(interpolateSharedMem(V, i, j, 0, 1, imax)) *
                            (V[idx] - V[idx_top]) / 2 -
                        fabs(interpolateSharedMem(V, i, j, 0, -1, imax)) *
                            (V[idx_bottom] - V[idx]) / 2);

  double term2 =
      (1 / _dx) * (interpolateSharedMem(U, i, j, 0, 1, imax) *
                       interpolateSharedMem(V, i, j, 1, 0, imax) -
                   interpolateSharedMem(U, i - 1, j, 0, 1, imax) *
                       interpolateSharedMem(V, i, j, -1, 0, imax)) +
      (_gamma / _dx) * (fabs(interpolateSharedMem(U, i, j, 0, 1, imax)) *
                            (V[idx] - V[idx_right]) / 2 -
                        fabs(interpolateSharedMem(U, i - 1, j, 0, 1, imax)) *
                            (V[idx_left] - V[idx]) / 2);

  return term1 + term2;
}

__device__ double Discretization::convection_T(const double *U, const double *V,
                                               const double *T, int i, int j) {
  int idx = _imax * j + i;
  int idx_right = _imax * j + (i + 1);
  int idx_left = _imax * j + (i - 1);
  double term1 =
      (1 / (2 * _dx)) * (U[idx] * (T[idx] + T[idx_right]) -
                         U[idx_left] * (T[idx_left] + T[idx])) +
      (_gamma / (2 * _dx)) * (fabs(U[idx]) * (T[idx] - T[idx_right]) -
                              fabs(U[idx_left]) * (T[idx_left] - T[idx]));

  int idx_top = _imax * (j + 1) + i;
  int idx_bottom = _imax * (j - 1) + i;
  double term2 =
      (1 / (2 * _dy)) * (V[idx] * (T[idx] + T[idx_top]) -
                         V[idx_bottom] * (T[idx_bottom] + T[idx])) +
      (_gamma / (2 * _dy)) * (fabs(V[idx]) * (T[idx] - T[idx_top]) -
                              fabs(V[idx_bottom]) * (T[idx_bottom] - T[idx]));
  return term1 + term2;
}

__device__ double Discretization::convection_TSharedMem(const double *U,
                                                        const double *V,
                                                        const double *T, int i,
                                                        int j, int imax) {
  int idx = imax * j + i;
  int idx_right = imax * j + (i + 1);
  int idx_left = imax * j + (i - 1);
  double term1 =
      (1 / (2 * _dx)) * (U[idx] * (T[idx] + T[idx_right]) -
                         U[idx_left] * (T[idx_left] + T[idx])) +
      (_gamma / (2 * _dx)) * (fabs(U[idx]) * (T[idx] - T[idx_right]) -
                              fabs(U[idx_left]) * (T[idx_left] - T[idx]));

  int idx_top = imax * (j + 1) + i;
  int idx_bottom = imax * (j - 1) + i;
  double term2 =
      (1 / (2 * _dy)) * (V[idx] * (T[idx] + T[idx_top]) -
                         V[idx_bottom] * (T[idx_bottom] + T[idx])) +
      (_gamma / (2 * _dy)) * (fabs(V[idx]) * (T[idx] - T[idx_top]) -
                              fabs(V[idx_bottom]) * (T[idx_bottom] - T[idx]));
  return term1 + term2;
}

__device__ double Discretization::diffusion(const double *A, int i, int j) {
  int idx = _imax * j + i;
  int idx_right = _imax * j + i + 1;
  int idx_left = _imax * j + i - 1;
  double term1 = (A[idx_right] - 2 * A[idx] + A[idx_left]) / (_dx * _dx);

  int idx_top = _imax * (j + 1) + i;
  int idx_bottom = _imax * (j - 1) + i;

  double term2 = (A[idx_top] - 2 * A[idx] + A[idx_bottom]) / (_dy * _dy);
  return term1 + term2;
}

__device__ double Discretization::diffusionSharedMem(const double *A, int i,
                                                     int j, int imax) {
  int idx = imax * j + i;
  int idx_right = imax * j + i + 1;
  int idx_left = imax * j + i - 1;
  double term1 = (A[idx_right] - 2 * A[idx] + A[idx_left]) / (_dx * _dx);

  int idx_top = imax * (j + 1) + i;
  int idx_bottom = imax * (j - 1) + i;

  double term2 = (A[idx_top] - 2 * A[idx] + A[idx_bottom]) / (_dy * _dy);
  return term1 + term2;
}

__device__ double Discretization::laplacian(const double *P, int i, int j) {
  int idx = _imax * j + i;
  int idx_right = _imax * j + i + 1;
  int idx_left = _imax * j + i - 1;
  int idx_top = _imax * (j + 1) + i;
  int idx_bottom = _imax * (j - 1) + i;

  double result = (P[idx_right] - 2.0 * P[idx] + P[idx_left]) / (_dx * _dx) +
                  (P[idx_top] - 2.0 * P[idx] + P[idx_bottom]) / (_dy * _dy);

  return result;
}

__device__ double Discretization::sor_helper(const double *P, int i, int j) {
  // int idx = _imax * j + i;
  int idx_right = _imax * j + i + 1;
  int idx_left = _imax * j + i - 1;
  int idx_top = _imax * (j + 1) + i;
  int idx_bottom = _imax * (j - 1) + i;

  double result = (P[idx_right] + P[idx_left]) / (_dx * _dx) +
                  (P[idx_top] + P[idx_bottom]) / (_dy * _dy);

  return result;
}

__device__ double Discretization::sor_helperSharedMem(const double *P, int i, int j, int imax) {
  // int idx = _imax * j + i;
  int idx_right = imax * j + i + 1;
  int idx_left = imax * j + i - 1;
  int idx_top = imax * (j + 1) + i;
  int idx_bottom = imax * (j - 1) + i;

  double result = (P[idx_right] + P[idx_left]) / (_dx * _dx) +
                  (P[idx_top] + P[idx_bottom]) / (_dy * _dy);

  return result;
}


__device__ double Discretization::interpolate(const double *A, int i, int j,
                                              int i_offset, int j_offset) {
  int idx = _imax * j + i;
  int idxOffset = _imax * (j + j_offset) + i + i_offset;

  return 0.5 * (A[idx] + A[idxOffset]);
}

__device__ double Discretization::interpolateSharedMem(const double *A, int i,
                                                       int j, int i_offset,
                                                       int j_offset, int imax) {
  int idx = imax * j + i;
  int idxOffset = imax * (j + j_offset) + i + i_offset;

  return 0.5 * (A[idx] + A[idxOffset]);
}
