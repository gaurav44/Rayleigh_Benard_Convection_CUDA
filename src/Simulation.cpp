#include "Simulation.hpp"
#include "Domain.hpp"

Simulation::Simulation(Fields *fields, Domain *domain)
    : _fields(fields), _domain(domain) {
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(
      (_domain->imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (_domain->jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  CHECK(cudaMalloc(&d_u_block_max, numBlocks.x * numBlocks.y * sizeof(double)));
  CHECK(cudaMalloc(&d_v_block_max, numBlocks.x * numBlocks.y * sizeof(double)));

  h_u_block_max = new double[numBlocks.x * numBlocks.y];
  h_v_block_max = new double[numBlocks.x * numBlocks.y];

  CHECK(cudaStreamCreate(&streamFU));
  CHECK(cudaStreamCreate(&streamGV));
  CHECK(cudaEventCreate(&eventFU));
  CHECK(cudaEventCreate(&eventGV));
}

void Simulation::calculate_dt() {
  double CFLu = 0.0;
  double CFLv = 0.0;
  double CFLnu = 0.0;
  double CFLt = 0.0;

  double dx2 = _domain->dx * _domain->dx;
  double dy2 = _domain->dy * _domain->dy;

  auto [u_max, v_max] =
      Dt_kernel(_fields->U, _fields->V, *_domain, d_u_block_max, d_v_block_max,
                h_u_block_max, h_v_block_max);

  CFLu = _domain->dx / u_max;
  CFLv = _domain->dy / v_max;

  CFLnu = (0.5 / _domain->nu) * (1.0 / (1.0 / dx2 + 1.0 / dy2));
  _domain->dt = std::min(CFLnu, std::min(CFLu, CFLv));

  CFLt = (0.5 / _domain->alpha) * (1.0 / (1.0 / dx2 + 1.0 / dy2));
  _domain->dt = std::min(_domain->dt, CFLt);

  _domain->dt = _domain->tau * _domain->dt;
}

void Simulation::calculate_temperature() {
  temperature_kernel(_fields->U, _fields->V, _fields->T, *_domain);
}

void Simulation::calculate_fluxes() {
  // F_kernel(_fields->U, _fields->V, _fields->T, _fields->F, *_domain);
  // G_kernel(_fields->U, _fields->V, _fields->T, _fields->G, *_domain);
  FandGKernel(_fields->U, _fields->V, _fields->F, _fields->G, _fields->T,
              *_domain);
}

void Simulation::calculate_rs() {
  RS_kernel(_fields->F, _fields->G, _fields->RS, *_domain);
}

void Simulation::calculate_velocities() {
  //U_kernel(_fields->U, _fields->F, _fields->P, *_domain);
  //V_kernel(_fields->V, _fields->G, _fields->P, *_domain);
  UV_kernel(_fields->U, _fields->V, _fields->F, _fields->G, _fields->P, *_domain);
}

Simulation::~Simulation() {
  CHECK(cudaFree(d_u_block_max));
  CHECK(cudaFree(d_v_block_max));
  CHECK(cudaStreamDestroy(streamFU));
  CHECK(cudaStreamDestroy(streamGV));
  CHECK(cudaEventDestroy(eventFU));
  CHECK(cudaEventDestroy(eventGV));
  free(h_u_block_max);
  free(h_v_block_max);
}
