#pragma once
#include "DataStructure.hpp"
#include "Domain.hpp"
#include <cmath>
class Discretization {
    public:
        // Calculating the value of convective part of U
        static double convection_u(const Matrix<double> &U,
                                   const Matrix<double> &V,
                                   const Domain& domain,
                                   int i, int j);
        // {
        //     double term1 =
        //         (1 / domain.dx) * (interpolate(U, i, j, 1, 0) * interpolate(U, i, j, 1, 0) -
        //                     interpolate(U, i, j, -1, 0) * interpolate(U, i, j, -1, 0)) +
        //         domain.gamma / domain.dx *
        //             (fabs(interpolate(U, i, j, 1, 0)) * (U(i, j) - U(i + 1, j)) / 2 -
        //             fabs(interpolate(U, i, j, -1, 0)) * (U(i - 1, j) - U(i, j)) / 2);

        //     double term2 =
        //         (1 / domain.dy) *
        //             (interpolate(V, i, j, 1, 0) * interpolate(U, i, j, 0, 1) -
        //             interpolate(V, i, j - 1, 1, 0) * interpolate(U, i, j, 0, -1)) +
        //         (domain.gamma / domain.dy) *
        //             (fabs(interpolate(V, i, j, 1, 0)) * (U(i, j) - U(i, j + 1)) / 2 -
        //             fabs(interpolate(V, i, j - 1, 1, 0)) * (U(i, j - 1) - U(i, j)) / 2);

        //     return term1 + term2;
        // }

        // Calculating the value of convective part of V
        static double convection_v(const Matrix<double> &U,
                                   const Matrix<double> &V,
                                   const Domain& domain,
                                   int i, int j);
        // {
        //     double term1 =
        //         (1 / domain.dy) * (interpolate(V, i, j, 0, 1) * interpolate(V, i, j, 0, 1) -
        //                     interpolate(V, i, j, 0, -1) * interpolate(V, i, j, 0, -1)) +
        //         (domain.gamma / domain.dy) *
        //             (fabs(interpolate(V, i, j, 0, 1)) * (V(i, j) - V(i, j + 1)) / 2 -
        //             fabs(interpolate(V, i, j, 0, -1)) * (V(i, j - 1) - V(i, j)) / 2);

        //     double term2 =
        //         (1 / domain.dx) *
        //             (interpolate(U, i, j, 0, 1) * interpolate(V, i, j, 1, 0) -
        //             interpolate(U, i - 1, j, 0, 1) * interpolate(V, i, j, -1, 0)) +
        //         (domain.gamma / domain.dx) *
        //             (fabs(interpolate(U, i, j, 0, 1)) * (V(i, j) - V(i + 1, j)) / 2 -
        //             fabs(interpolate(U, i - 1, j, 0, 1)) * (V(i - 1, j) - V(i, j)) / 2);

        //     return term1 + term2;
        // }

        // Calculating the value of convective part of T
        static constexpr double convection_T(const Matrix<double> &U,
                                   const Matrix<double> &V,
                                   const Matrix<double> &T,
                                   const Domain& domain,
                                   int i, int j);
        // {
        //     double term1 =
        //         (1 / (2 * domain.dx)) * (U(i, j) * (T(i, j) + T(i + 1, j)) -
        //                             U(i - 1, j) * (T(i - 1, j) + T(i, j))) +
        //         (domain.gamma / (2 * domain.dx)) * (fabs(U(i, j)) * (T(i, j) - T(i + 1, j)) -
        //                                 fabs(U(i - 1, j)) * (T(i - 1, j) - T(i, j)));

        //     double term2 =
        //         (1 / (2 * domain.dy)) * (V(i, j) * (T(i, j) + T(i, j + 1)) -
        //                             V(i, j - 1) * (T(i, j - 1) + T(i, j))) +
        //         (domain.gamma / (2 * domain.dy)) * (fabs(V(i, j)) * (T(i, j) - T(i, j + 1)) -
        //                                 fabs(V(i, j - 1)) * (T(i, j - 1) - T(i, j)));

        //     return term1 + term2;
        // };

        // Using the same for calculating diffusive part of U and V
        static constexpr double diffusion(const Matrix<double> &A,
                                const Domain& domain,
                                int i, int j);
        // {
        //     double term1 = (A(i + 1, j) - 2 * A(i, j) + A(i - 1, j)) / (domain.dx * domain.dx);
        //     double term2 = (A(i, j + 1) - 2 * A(i, j) + A(i, j - 1)) / (domain.dy * domain.dy);

        //     return term1 + term2;
        // }

        // Calculating the laplacian part of the equation

        static double laplacian(const Matrix<double> &P,
                                const Domain& domain,
                                int i, int j);
        // {
        //     double result = (P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) / (domain.dx * domain.dx) +
        //                     (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) / (domain.dy * domain.dy);
        //     return result;
        // }

        // Calculating the SOR Helper
        static double sor_helper(const Matrix<double> &P,
                                 const Domain& domain,
                                 int i, int j);
        // {
        //     double result = (P(i + 1, j) + P(i - 1, j)) / (domain.dx * domain.dx) +
        //                     (P(i, j + 1) + P(i, j - 1)) / (domain.dy * domain.dy);
        //     return result;
        // }

        static double interpolate(const Matrix<double> &A,
                                  int i, int j,
                                  int i_offset, int j_offset);
        // {
        //     return (A(i, j) + A(i + i_offset, j + j_offset)) / 2;
        // }
};
