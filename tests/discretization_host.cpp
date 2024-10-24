#include "discretization_host.hpp"
int DiscretizationHost::_imax;
int DiscretizationHost::_jmax;
double  DiscretizationHost::_dx;
double DiscretizationHost::_dy;
double DiscretizationHost::_gamma;

DiscretizationHost::DiscretizationHost(int imax, int jmax, double dx, double dy, double gamma) {
    _imax = imax;
    _jmax = jmax;
    _dx = dx;
    _dy = dy;
    _gamma = gamma;
}

double DiscretizationHost::convection_u(const double* U,
                                        const double* V,
                                        int i, int j) {

    int idx = j * _imax + i;
    int idxRight = idx + 1;
    int idxLeft = idx - 1;
    int idxTop = idx + _imax;
    int idxBottom = idx - _imax;

    double term1 = (1 / _dx) * (interpolate(U, i, j, 1, 0) * interpolate(U, i, j, 1, 0) 
                                    - interpolate(U, i, j, -1, 0) * interpolate(U, i, j, -1, 0)) +
                   (_gamma / _dx) * (fabs(interpolate(U, i, j, 1, 0)) * (U[idx] - U[idxRight]) / 2 
                                          - fabs(interpolate(U, i, j, -1, 0)) * (U[idxLeft] - U[idx]) / 2);

    double term2 = (1 / _dy) * (interpolate(V, i, j, 1, 0) * interpolate(U, i, j, 0, 1)
                                   -  interpolate(V, i, j - 1, 1, 0) * interpolate(U, i, j, 0, -1)) +
                   (_gamma / _dy) * (fabs(interpolate(V, i, j, 1, 0)) * (U[idx] - U[idxTop]) / 2
                                              -  fabs(interpolate(V, i, j - 1, 1, 0)) * (U[idxBottom] - U[idx]) / 2);
    return term1 + term2;

}

double DiscretizationHost::convection_v(const double *U,
                                        const double *V,
                                        int i, int j) {
    int idx = j * _imax + i;
    int idxRight = idx + 1;
    int idxLeft = idx - 1;
    int idxTop = idx + _imax;
    int idxBottom = idx - _imax;

    double term1 =
        (1 / _dy) * (interpolate(V, i, j, 0, 1) * interpolate(V, i, j, 0, 1) -
                    interpolate(V, i, j, 0, -1) * interpolate(V, i, j, 0, -1)) +
        (_gamma / _dy) *
            (fabs(interpolate(V, i, j, 0, 1)) * (V[idx] - V[idxTop]) / 2 -
            fabs(interpolate(V, i, j, 0, -1)) * (V[idxBottom] - V[idx]) / 2);

    double term2 =
        (1 / _dx) *
            (interpolate(U, i, j, 0, 1) * interpolate(V, i, j, 1, 0) -
            interpolate(U, i - 1, j, 0, 1) * interpolate(V, i, j, -1, 0)) +
        (_gamma / _dx) *
            (fabs(interpolate(U, i, j, 0, 1)) * (V[idx] - V[idxRight]) / 2 -
            fabs(interpolate(U, i - 1, j, 0, 1)) * (V[idxLeft] - V[idx]) / 2);

    return term1 + term2;
}

 double DiscretizationHost::convection_T(const double *U,
                                         const double *V,
                                         const double *T,
                                         int i, int j) {
    int idx = j * _imax + i;
    int idxRight = idx + 1;
    int idxLeft = idx - 1;
    int idxTop = idx + _imax;
    int idxBottom = idx - _imax;

    double term1 =
        (1 / (2 * _dx)) * (U[idx] * (T[idx] + T[idxRight]) -
                            U[idxLeft] * (T[idxLeft] + T[idx])) +
        (_gamma / (2 * _dx)) * (fabs(U[idx]) * (T[idx] - T[idxRight]) -
                                fabs(U[idxLeft]) * (T[idxLeft] - T[idx]));

    double term2 =
        (1 / (2 * _dy)) * (V[idx] * (T[idx] + T[idxTop]) -
                            V[idxBottom] * (T[idxBottom] + T[idx])) +
        (_gamma / (2 * _dy)) * (fabs(V[idx]) * (T[idx] - T[idxTop]) -
                                fabs(V[idxBottom]) * (T[idxBottom] - T[idx]));

    return term1 + term2;
};

 double DiscretizationHost::diffusion(const double *A,
                                      int i, int j) {
    int idx = j * _imax + i;
    int idxRight = idx + 1;
    int idxLeft = idx - 1;
    int idxTop = idx + _imax;
    int idxBottom = idx - _imax;

    double term1 = (A[idxRight] - 2 * A[idx] + A[idxLeft]) / (_dx * _dx);
    double term2 = (A[idxTop] - 2 * A[idx] + A[idxBottom]) / (_dy * _dy);

    return term1 + term2;
}

double DiscretizationHost::laplacian(const double *P,
                                     int i, int j) {
    
    int idx = j * _imax + i;
    int idxRight = idx + 1;
    int idxLeft = idx - 1;
    int idxTop = idx + _imax;
    int idxBottom = idx - _imax;

    double result = (P[idxRight] - 2.0 * P[idx] + P[idxLeft]) / (_dx * _dx) +
                    (P[idxTop] - 2.0 * P[idx] + P[idxBottom]) / (_dy * _dy);

    return result;
}

double DiscretizationHost::sor_helper(const double *P,
                                      int i, int j) {
    int idx = j * _imax + i;
    int idxRight = idx + 1;
    int idxLeft = idx - 1;
    int idxTop = idx + _imax;
    int idxBottom = idx - _imax;
    double result = (P[idxRight] + P[idxLeft]) / (_dx * _dx) +
                    (P[idxTop] + P[idxBottom]) / (_dy * _dy);
    
    return result;
}

double DiscretizationHost::interpolate(const double *A,
                                       int i, int j,
                                       int i_offset, int j_offset) {
    int idx = _imax * j + i;
    int idxOffset = idx + _imax*j_offset + i_offset;
    return (A[idx] + A[idxOffset]) / 2;
}
