#include "simulation.hpp"
#include "domain.hpp"

Simulation::Simulation(Fields *fields, Domain *domain)
    : _fields(fields), _domain(domain) {
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(
      (_domain->imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (_domain->jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  CHECK(cudaMalloc(&d_uBlockMax, numBlocks.x * numBlocks.y * sizeof(double)));
  CHECK(cudaMalloc(&d_vBlockMax, numBlocks.x * numBlocks.y * sizeof(double)));

  h_uBlockMax = new double[numBlocks.x * numBlocks.y];
  h_vBlockMax = new double[numBlocks.x * numBlocks.y];
}

void Simulation::calculateTimeStep() {
  double CFLu = 0.0;
  double CFLv = 0.0;
  double CFLnu = 0.0;
  double CFLt = 0.0;

  double dx2 = _domain->dx * _domain->dx;
  double dy2 = _domain->dy * _domain->dy;

  auto [u_max, v_max] =
      TimestepKernels::calculateUVMaxKernel(_fields->U, _fields->V, *_domain, d_uBlockMax, d_vBlockMax,
                h_uBlockMax, h_vBlockMax);

  CFLu = _domain->dx / u_max;
  CFLv = _domain->dy / v_max;

  double multiplier = (1.0 / (1.0 / dx2 + 1.0 / dy2));

  CFLnu = (0.5 / _domain->nu) * multiplier;//(1.0 / (1.0 / dx2 + 1.0 / dy2));
  _domain->dt = std::min(CFLnu, std::min(CFLu, CFLv));

  CFLt = (0.5 / _domain->alpha) * multiplier;//(1.0 / (1.0 / dx2 + 1.0 / dy2));
  _domain->dt = std::min(_domain->dt, CFLt);

  _domain->dt = _domain->tau * _domain->dt;
}

void Simulation::calculateTemperature() {
  TemperatureKernels::calculateTemperatureKernel(_fields->U, _fields->V, _fields->T, *_domain);
}

void Simulation::calculateFluxes() {
  FluxesKernels::calculateFluxesKernel(_fields->U, _fields->V, _fields->F, _fields->G, _fields->T,
              *_domain);
}

void Simulation::calculateRightHandSide() {
  RightHandSideKernels::calculateRightHandSideKernel(_fields->F, _fields->G, _fields->RS, *_domain);
}

void Simulation::calculateVelocities() {
  VelocityKernels::calculateVelocitiesKernel(_fields->U, _fields->V, _fields->F, _fields->G, _fields->P, *_domain);
}

Simulation::~Simulation() {
  CHECK(cudaFree(d_uBlockMax));
  CHECK(cudaFree(d_vBlockMax));
 
  free(h_uBlockMax);
  free(h_vBlockMax);
}
