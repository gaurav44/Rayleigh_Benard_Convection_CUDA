#include "Boundary.hpp"
#include "DataStructure.hpp"
#include "Domain.hpp"
#include "Fields.hpp"
#include "PressureSolver.hpp"
#include "Simulation.hpp"
#include <iostream>
#include <unistd.h>

#define BLOCK_SIZE = 8;

int main() {
  Domain domain;

  domain.xlength = 8.5;
  domain.ylength = 1;
  domain.nu = 0.0296;                   // Kinematic Viscosity
  domain.Re = 1 / domain.nu;            // Reynold's number
  domain.alpha = 0.00000237;            // Thermal diffusivity
  domain.Pr = domain.nu / domain.alpha; // Prandtl number
  domain.beta =
      0.00179; // Coefficient of thermal expansion (used in Boussinesq approx)
  domain.tau = 0.5;   // Safety factor for timestep
  domain.gamma = 0.5; // Donor-cell scheme factor (will be used in convection)
  domain.GX = 0;
  domain.GY = -9.81; // Gravitational acceleration
  domain.imax = 85;  // grid points in x
  domain.jmax = 18;  // grid points in y
  domain.dx = domain.xlength / domain.imax;
  domain.dy = domain.ylength / domain.jmax;
  domain.dt = 0.05; // Timestep

  Fields fields;
  fields.U = Matrix(domain.imax + 2, domain.jmax + 2);
  fields.V = Matrix(domain.imax + 2, domain.jmax + 2);
  fields.F = Matrix(domain.imax + 2, domain.jmax + 2);
  fields.G = Matrix(domain.imax + 2, domain.jmax + 2);
  fields.P = Matrix(domain.imax + 2, domain.jmax + 2);
  fields.T = Matrix(domain.imax + 2, domain.jmax + 2, 293.0);
  fields.T_old = Matrix(domain.imax + 2, domain.jmax + 2, 293.0);
  fields.RS = Matrix(domain.imax + 2, domain.jmax + 2, 0.0);

  double t_end = 55000;
  double omg = 1.7;     // SOR relaxation factor
  double eps = 0.00001; // Tolerance for SOR

  int itermax = 500; // Maximum iterations for SOR

  double Th = 294.78;
  double Tc = 291.20;

  // Apply Boundaries
  fields.F.copyToDevice();
  fields.G.copyToDevice();
  fields.T.copyToDevice();
  fields.U.copyToDevice();
  fields.V.copyToDevice();
  fields.P.copyToDevice();

  Boundary::apply_boundaries(fields, domain, Th, Tc);
  Boundary::apply_pressure(fields.P, domain);

  double t = 0;
  int timestep = 0;
  // Time loop
  while (t < t_end) {
    Simulation::calculate_dt(domain, fields);
    
    Simulation::calculate_temperature(fields.U, fields.V, fields.T, domain);
    
    Simulation::calculate_fluxes(fields.U, fields.V, fields.T, fields.F,
                                 fields.G, domain);
                                 
    Simulation::calculate_rs(fields.F, fields.G, fields.RS, domain);
  
    int iter = 0;
    double res = 10;

    while (res > eps) {
      if (iter >= itermax) {
        std::cout << "Pressure solver not converged\n";
        std::cout << "dt: " << domain.dt << "Time: "
                  << " residual:" << res << " iterations: " << iter << "\n";
        break;
      }
      Boundary::apply_pressure(fields.P, domain);
    
      res =
          PressureSolver::calculate_pressure(fields.P, fields.RS, domain, omg);
      iter++;
    }
    Simulation::calculate_velocities(fields.U, fields.V, fields.F, fields.G,
                                     fields.P, domain);
  
    Boundary::apply_boundaries(fields, domain, Th, Tc);

    if (timestep % 1000 == 0) {
      if (timestep % 15000 == 0) {
        fields.T.copyToHost();
        fields.T.printField(timestep);
      }
      std::cout << "dt: " << domain.dt << "Time: " << t << " residual:" << res
                << " iterations: " << iter << "\n";
    }
    t = t + domain.dt;
    timestep++;
  }
}
