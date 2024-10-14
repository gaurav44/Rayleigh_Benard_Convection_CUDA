#pragma once

#include "DataStructure.hpp"
#include "Discretization.hpp"
#include "Domain.hpp"
#include "Fields.hpp"
#include <cmath>

class Simulation {
public:
  static void calculate_dt(Domain &domain, const Fields &fields) {
    double CFLu = 0.0;
    double CFLv = 0.0;
    double CFLnu = 0.0;
    double CFLt = 0.0;

    double dx2 = domain.dx * domain.dx;
    double dy2 = domain.dy * domain.dy;

    double u_max = 0;
    double v_max = 0;

    for (int i = 1; i < domain.imax + 1; i++) {
      for (int j = 1; j < domain.jmax + 1; j++) {
        u_max = std::max(u_max, fabs(fields.U(i, j)));
        v_max = std::max(v_max, fabs(fields.V(i, j)));
      }
    }

    CFLu = domain.dx / u_max;
    CFLv = domain.dy / v_max;

    CFLnu = (0.5 / domain.nu) * (1.0 / (1.0 / dx2 + 1.0 / dy2));
    domain.dt = std::min(CFLnu, std::min(CFLu, CFLv));

    CFLt = (0.5 / domain.alpha) * (1.0 / (1.0 / dx2 + 1.0 / dy2));
    domain.dt = std::min(domain.dt, CFLt);

    domain.dt = domain.tau * domain.dt;
  };

  static void calculate_temperature(const Matrix &U, const Matrix &V, Matrix &T,
                                    const Domain &domain);

  static void calculate_fluxes(const Matrix &U, const Matrix &V,
                               const Matrix &T, Matrix &F, Matrix &G,
                               const Domain &domain);

  static void calculate_rs(const Matrix &F, const Matrix &G, Matrix &RS,
                           const Domain &domain);

  static void calculate_velocities(Matrix &U, Matrix &V, const Matrix &F,
                                   const Matrix &G, const Matrix &P,
                                   const Domain &domain);
};
