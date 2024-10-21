#include "discretization.hpp"
int Discretization::_imax;
int Discretization::_jmax;
double  Discretization::_dx;
double Discretization::_dy;
double Discretization::_gamma;

Discretization::Discretization(int imax, int jmax, double dx, double dy, double gamma) {
    _imax = imax;
    _jmax = jmax;
    _dx = dx;
    _dy = dy;
    _gamma = gamma;
}

double Discretization::convection_u(const Matrix &U,
                                    const Matrix &V,
                                    const Domain& domain,
                                    int i, int j) {

    double term1 = (1 / _dx) * (interpolate(U, i, j, 1, 0) * interpolate(U, i, j, 1, 0) 
                                    - interpolate(U, i, j, -1, 0) * interpolate(U, i, j, -1, 0)) +
                   (_gamma / _dx) * (fabs(interpolate(U, i, j, 1, 0)) * (U(i, j) - U(i + 1, j)) / 2 
                                          - fabs(interpolate(U, i, j, -1, 0)) * (U(i - 1, j) - U(i, j)) / 2);

    double term2 = (1 / _dy) * (interpolate(V, i, j, 1, 0) * interpolate(U, i, j, 0, 1)
                                   -  interpolate(V, i, j - 1, 1, 0) * interpolate(U, i, j, 0, -1)) +
                   (_gamma / _dy) * (fabs(interpolate(V, i, j, 1, 0)) * (U(i, j) - U(i, j + 1)) / 2
                                              -  fabs(interpolate(V, i, j - 1, 1, 0)) * (U(i, j - 1) - U(i, j)) / 2);
            return term1 + term2;

}

double Discretization::convection_v(const Matrix &U,
                                    const Matrix &V,
                                    const Domain& domain,
                                    int i, int j) {
            double term1 =
                (1 / _dy) * (interpolate(V, i, j, 0, 1) * interpolate(V, i, j, 0, 1) -
                            interpolate(V, i, j, 0, -1) * interpolate(V, i, j, 0, -1)) +
                (_gamma / _dy) *
                    (fabs(interpolate(V, i, j, 0, 1)) * (V(i, j) - V(i, j + 1)) / 2 -
                    fabs(interpolate(V, i, j, 0, -1)) * (V(i, j - 1) - V(i, j)) / 2);

            double term2 =
                (1 / _dx) *
                    (interpolate(U, i, j, 0, 1) * interpolate(V, i, j, 1, 0) -
                    interpolate(U, i - 1, j, 0, 1) * interpolate(V, i, j, -1, 0)) +
                (_gamma / _dx) *
                    (fabs(interpolate(U, i, j, 0, 1)) * (V(i, j) - V(i + 1, j)) / 2 -
                    fabs(interpolate(U, i - 1, j, 0, 1)) * (V(i - 1, j) - V(i, j)) / 2);

            return term1 + term2;
}

 double Discretization::convection_T(const Matrix &U,
                                    const Matrix &V,
                                    const Matrix &T,
                                    const Domain& domain,
                                    int i, int j) {
            double term1 =
                (1 / (2 * _dx)) * (U(i, j) * (T(i, j) + T(i + 1, j)) -
                                    U(i - 1, j) * (T(i - 1, j) + T(i, j))) +
                (_gamma / (2 * _dx)) * (fabs(U(i, j)) * (T(i, j) - T(i + 1, j)) -
                                        fabs(U(i - 1, j)) * (T(i - 1, j) - T(i, j)));

            double term2 =
                (1 / (2 * _dy)) * (V(i, j) * (T(i, j) + T(i, j + 1)) -
                                    V(i, j - 1) * (T(i, j - 1) + T(i, j))) +
                (_gamma / (2 * _dy)) * (fabs(V(i, j)) * (T(i, j) - T(i, j + 1)) -
                                        fabs(V(i, j - 1)) * (T(i, j - 1) - T(i, j)));

            return term1 + term2;
};

 double Discretization::diffusion(const Matrix &A,
                                            const Domain& domain,
                                            int i, int j) {
    double term1 = (A(i + 1, j) - 2 * A(i, j) + A(i - 1, j)) / (_dx * _dx);
    double term2 = (A(i, j + 1) - 2 * A(i, j) + A(i, j - 1)) / (_dy * _dy);

    return term1 + term2;
}

double Discretization::laplacian(const Matrix &P,
                                 const Domain& domain,
                                 int i, int j) {
    double result = (P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) / (_dx * _dx) +
                    (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) / (_dy * _dy);

    return result;
}

double Discretization::sor_helper(const Matrix &P,
                                  const Domain& domain,
                                  int i, int j) {
    double result = (P(i + 1, j) + P(i - 1, j)) / (_dx * _dx) +
                    (P(i, j + 1) + P(i, j - 1)) / (_dy * _dy);
    
    return result;
}

double Discretization::interpolate(const Matrix &A,
                                   int i, int j,
                                   int i_offset, int j_offset) {
    return (A(i, j) + A(i + i_offset, j + j_offset)) / 2;
}
