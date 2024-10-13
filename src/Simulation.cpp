#include "Simulation.hpp"

Simulation::Simulation(Fields* fields) : _fields(fields) {}

void Simulation::calculate_dt(Domain &domain) {
  double CFLu = 0.0;
  double CFLv = 0.0;
  double CFLnu = 0.0;
  double CFLt = 0.0;

  double dx2 = domain.dx * domain.dx;
  double dy2 = domain.dy * domain.dy;

  // double u_max = 0;
  // double v_max = 0;

  auto [u_max, v_max] = Dt_kernel(_fields->U, _fields->V, domain);

  // for (int i = 1; i < domain.imax + 1; i++) {
  //   for (int j = 1; j < domain.jmax + 1; j++) {
  //     u_max = std::max(u_max, fabs(fields.U(i, j)));
  //     v_max = std::max(v_max, fabs(fields.V(i, j)));
  //   }
  // }

  CFLu = domain.dx / u_max;
  CFLv = domain.dy / v_max;

  CFLnu = (0.5 / domain.nu) * (1.0 / (1.0 / dx2 + 1.0 / dy2));
  domain.dt = std::min(CFLnu, std::min(CFLu, CFLv));

  CFLt = (0.5 / domain.alpha) * (1.0 / (1.0 / dx2 + 1.0 / dy2));
  domain.dt = std::min(domain.dt, CFLt);

  domain.dt = domain.tau * domain.dt;
}

void Simulation::calculate_temperature(const Domain &domain) {

  // Matrix T_old = T;
  // T_old.copyToDevice();

  // temperature_kernel(U, V, T, T_old, domain);
  temperature_kernel(_fields->U, _fields->V, _fields->T, domain);
  //      T.d_container, U.d_container, V.d_container, domain.alpha, domain.dt,
  // domain.imax, domain.jmax);

  // for (int i = 1; i < domain.imax + 1; i++) {
  //   for (int j = 1; j < domain.jmax + 1; j++) {
  //     T(i, j) =
  //         T_old(i, j) +
  //         domain.dt *
  //             (domain.alpha * Discretization::diffusion(T_old, domain, i, j)
  //             -
  //              Discretization::convection_T(U, V, T_old, domain, i, j));
  //   }
  // }
}

void Simulation::calculate_fluxes(const Domain &domain) {
  F_kernel(_fields->U, _fields->V, _fields->T, _fields->F, domain);
  G_kernel(_fields->U, _fields->V, _fields->T, _fields->G, domain);

  //  for (int i = 1; i < domain.imax; i++) {
  //    for (int j = 1; j < domain.jmax + 1; j++) {
  //      F(i, j) =
  //          U(i, j) +
  //          domain.dt * (domain.nu * Discretization::diffusion(U, domain, i,
  //          j)
  //          -
  //                       Discretization::convection_u(U, V, domain, i, j)) -
  //          (domain.beta * domain.dt / 2 * (T(i, j) + T(i + 1, j))) *
  //          domain.GX;
  //    }
  //  }

  //  for (int i = 1; i < domain.imax + 1; i++) {
  //    for (int j = 1; j < domain.jmax; j++) {
  //      G(i, j) =
  //          V(i, j) +
  //          domain.dt * (domain.nu * Discretization::diffusion(V, domain, i,
  //          j)
  //          -
  //                       Discretization::convection_v(U, V, domain, i, j)) -
  //          (domain.beta * domain.dt / 2 * (T(i, j) + T(i, j + 1))) *
  //          domain.GY;
  //    }
  //  }
}

void Simulation::calculate_rs(const Domain &domain) {
  RS_kernel(_fields->F, _fields->G, _fields->RS, domain);

  // for (int i = 1; i < domain.imax + 1; i++) {
  //   for (int j = 1; j < domain.jmax + 1; j++) {
  //     double term1 = (F(i, j) - F(i - 1, j)) / domain.dx;
  //     double term2 = (G(i, j) - G(i, j - 1)) / domain.dy;
  //     RS(i, j) = (term1 + term2) / domain.dt;
  //   }
  // }
}

void Simulation::calculate_velocities(const Domain &domain) {
  U_kernel(_fields->U, _fields->F, _fields->P, domain);
  V_kernel(_fields->V, _fields->G, _fields->P, domain);
  // for (int i = 1; i < domain.imax; i++) {
  //   for (int j = 1; j < domain.jmax + 1; j++) {
  //     U(i, j) = F(i, j) - domain.dt * (P(i + 1, j) - P(i, j)) / domain.dx;
  //   }
  // }

  // for (int i = 1; i < domain.imax + 1; i++) {
  //   for (int j = 1; j < domain.jmax; j++) {
  //     V(i, j) = G(i, j) - domain.dt * (P(i, j + 1) - P(i, j)) / domain.dy;
  //   }
  // }
}
