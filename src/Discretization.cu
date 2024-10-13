#include "Discretization.hpp"

Discretization::Discretization(int imax, int jmax, double dx, double dy,
                               double gamma) {
  //_imax = imax;
  //_jmax = jmax;
  //_dx = dx;
  //_dy = dy;
  //_gamma = gamma;
  cudaMemcpyToSymbol(_imax, &imax, sizeof(int));
  cudaMemcpyToSymbol(_jmax, &jmax, sizeof(int));
  cudaMemcpyToSymbol(_dx, &dx, sizeof(double));
  cudaMemcpyToSymbol(_dy, &dy, sizeof(double));
  cudaMemcpyToSymbol(_gamma, &gamma, sizeof(double));
}

double Discretization::convection_u(const Matrix &U, const Matrix &V,
                                    const Domain &domain, int i, int j) {

  double term1 =
      (1 / domain.dx) *
          (interpolate(U, i, j, 1, 0) * interpolate(U, i, j, 1, 0) -
           interpolate(U, i, j, -1, 0) * interpolate(U, i, j, -1, 0)) +
      (domain.gamma / domain.dx) *
          (fabs(interpolate(U, i, j, 1, 0)) * (U(i, j) - U(i + 1, j)) / 2 -
           fabs(interpolate(U, i, j, -1, 0)) * (U(i - 1, j) - U(i, j)) / 2);

  double term2 =
      (1 / domain.dy) *
          (interpolate(V, i, j, 1, 0) * interpolate(U, i, j, 0, 1) -
           interpolate(V, i, j - 1, 1, 0) * interpolate(U, i, j, 0, -1)) +
      (domain.gamma / domain.dy) *
          (fabs(interpolate(V, i, j, 1, 0)) * (U(i, j) - U(i, j + 1)) / 2 -
           fabs(interpolate(V, i, j - 1, 1, 0)) * (U(i, j - 1) - U(i, j)) / 2);
  return term1 + term2;
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

double Discretization::convection_v(const Matrix &U, const Matrix &V,
                                    const Domain &domain, int i, int j) {
  double term1 =
      (1 / domain.dy) *
          (interpolate(V, i, j, 0, 1) * interpolate(V, i, j, 0, 1) -
           interpolate(V, i, j, 0, -1) * interpolate(V, i, j, 0, -1)) +
      (domain.gamma / domain.dy) *
          (fabs(interpolate(V, i, j, 0, 1)) * (V(i, j) - V(i, j + 1)) / 2 -
           fabs(interpolate(V, i, j, 0, -1)) * (V(i, j - 1) - V(i, j)) / 2);

  double term2 =
      (1 / domain.dx) *
          (interpolate(U, i, j, 0, 1) * interpolate(V, i, j, 1, 0) -
           interpolate(U, i - 1, j, 0, 1) * interpolate(V, i, j, -1, 0)) +
      (domain.gamma / domain.dx) *
          (fabs(interpolate(U, i, j, 0, 1)) * (V(i, j) - V(i + 1, j)) / 2 -
           fabs(interpolate(U, i - 1, j, 0, 1)) * (V(i - 1, j) - V(i, j)) / 2);

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

double Discretization::convection_T(const Matrix &U, const Matrix &V,
                                    const Matrix &T, const Domain &domain,
                                    int i, int j) {
  double term1 =
      (1 / (2 * domain.dx)) * (U(i, j) * (T(i, j) + T(i + 1, j)) -
                               U(i - 1, j) * (T(i - 1, j) + T(i, j))) +
      (domain.gamma / (2 * domain.dx)) *
          (fabs(U(i, j)) * (T(i, j) - T(i + 1, j)) -
           fabs(U(i - 1, j)) * (T(i - 1, j) - T(i, j)));

  double term2 =
      (1 / (2 * domain.dy)) * (V(i, j) * (T(i, j) + T(i, j + 1)) -
                               V(i, j - 1) * (T(i, j - 1) + T(i, j))) +
      (domain.gamma / (2 * domain.dy)) *
          (fabs(V(i, j)) * (T(i, j) - T(i, j + 1)) -
           fabs(V(i, j - 1)) * (T(i, j - 1) - T(i, j)));

  return term1 + term2;
};

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

double Discretization::diffusion(const Matrix &A, const Domain &domain, int i,
                                 int j) {
  double term1 =
      (A(i + 1, j) - 2 * A(i, j) + A(i - 1, j)) / (domain.dx * domain.dx);
  double term2 =
      (A(i, j + 1) - 2 * A(i, j) + A(i, j - 1)) / (domain.dy * domain.dy);

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

double Discretization::laplacian(const Matrix &P, const Domain &domain, int i,
                                 int j) {
  double result =
      (P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) / (domain.dx * domain.dx) +
      (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) / (domain.dy * domain.dy);

  return result;
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

double Discretization::sor_helper(const Matrix &P, const Domain &domain, int i,
                                  int j) {
  double result = (P(i + 1, j) + P(i - 1, j)) / (domain.dx * domain.dx) +
                  (P(i, j + 1) + P(i, j - 1)) / (domain.dy * domain.dy);

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

double Discretization::interpolate(const Matrix &A, int i, int j, int i_offset,
                                   int j_offset) {
  return (A(i, j) + A(i + i_offset, j + j_offset)) / 2;
}

__device__ double Discretization::interpolate(const double *A, int i, int j,
                                              int i_offset, int j_offset) {
  int idx = _imax * j + i;
  int idxOffset = _imax * (j + j_offset) + i + i_offset;

  return 0.5 * (A[idx] + A[idxOffset]);
}
