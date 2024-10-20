#include "Boundary.hpp"
#include "DataStructure.hpp"
#include "Discretization.hpp"
#include "Domain.hpp"
#include "Fields.hpp"
#include "PressureSolver.hpp"
#include "Simulation.hpp"
#include <iostream>
#include <unistd.h>
#include <chrono>

int main() {
  Domain domain;

  domain.readDomainParameters("domain.txt");

  Fields fields(domain.imax + 2, domain.jmax + 2, 293.0);

  Discretization disc(domain.imax + 2, domain.jmax + 2, domain.dx, domain.dy,
                      domain.gamma);

  Simulation sim(&fields, &domain);
  sim.copyAllToDevice();
  Boundary boundary(&fields, &domain, 294.78, 291.20); // T_hot, T_cold -> for the top and bottom boundaries
  PressureSolver presSolver(&domain);

  boundary.apply_boundaries();
  boundary.apply_pressure();

  double t = 0;
  double t_end = 1000;
  int timestep = 0;
  auto start = std::chrono::high_resolution_clock::now();
  // Time loop
  while (t < t_end) {
    sim.calculate_dt();

    sim.calculate_temperature();
    
    sim.calculate_fluxes();

    sim.calculate_rs();

    int iter = 0;
    double res = 10.0;

    while (res > PressureSolver::eps) {
      if (iter >= PressureSolver::itermax) {
        std::cout << "Pressure solver not converged\n";
        std::cout << "dt: " << domain.dt << "Time: "
                  << " residual:" << res << " iterations: " << iter << "\n";
        break;
      }
      boundary.apply_pressure();

      res = presSolver.calculate_pressure(sim.getP(), sim.getRS(), domain);
      iter++;
    }
    sim.calculate_velocities();

    boundary.apply_boundaries();

    if (timestep % 1000 == 0) {
      // if (timestep % 15000 == 0) {
      //   sim.getT().copyToHost();
      //   sim.getT().printField(timestep);
      // }
      std::cout << "dt: " << domain.dt << "Time: " << t << " residual:" << res
                << " iterations: " << iter << "\n";
    }
    t = t + domain.dt;
    timestep++;
  }
  // Stop measuring time
  auto end = std::chrono::high_resolution_clock::now();
  // Calculate the duration
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " milliseconds\n";
}
