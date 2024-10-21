#include "boundary.hpp"
#include "datastructure.hpp"
#include "domain.hpp"
#include "fields.hpp"
#include "pressure_solver.hpp"
#include "simulation.hpp"
#include <iostream>
#include <chrono>

int main() {
  Domain domain;

  domain.readDomainParameters("domain.txt");

  Fields fields(domain.imax + 2, domain.jmax + 2, 293.0);

  Discretization disc(domain.imax + 2, domain.jmax + 2, domain.dx, domain.dy,
                      domain.gamma);

  Simulation sim(&fields, &domain);
  Boundary boundary(&fields, &domain, 294.78, 291.20); // T_hot, T_cold -> for the top and bottom boundaries
  PressureSolver presSolver(&domain);

  double t_end = 1000;
  double omg = 1.7;     // SOR relaxation factor
  double eps = 0.00001; // Tolerance for SOR

  int itermax = 500; // Maximum iterations for SOR

  double Th = 294.78;
  double Tc = 291.20;

  // Apply Boundaries
  boundary.applyBoundaries();
  boundary.applyPressureBoundary();
  double t = 0;
  int timestep = 0;

  auto start = std::chrono::high_resolution_clock::now();
  // Time loop
  while (t < t_end) {
    sim.calculateTimestep();

    sim.calculateTemperature();

    sim.calculateFluxes();

    sim.calculateRightHandSide();

    int iter = 0;
    double res = 10;

    while (res > PressureSolver::eps) {
      if (iter >= PressureSolver::itermax) {
        std::cout << "Pressure solver not converged\n";
        std::cout << "dt: " << domain.dt << "Time: "
                  << " residual:" << res << " iterations: " << iter << "\n";
        break;
      }
      boundary.applyPressureBoundary();
      res =
          presSolver.calculatePressure(fields.P, fields.RS);
      iter++;
    }

    sim.calculateVelocities();

    boundary.applyBoundaries();

    if (timestep % 1000 == 0)
      std::cout << "dt: " << domain.dt << "Time: " << t << " residual:" << res
                << " iterations: " << iter << "\n";
    t = t + domain.dt;
    timestep++;
  }
  // Measure execution time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Execution time: " << duration.count() << " ms\n";
}
