#include "PressureSolver.hpp"

PressureSolver::PressureSolver(Domain *domain) : _domain(domain) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(
      (_domain->imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (_domain->jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  CHECK(cudaMalloc(&d_rloc, numBlocks.x * numBlocks.y * sizeof(double)));
}

PressureSolver::~PressureSolver() { CHECK(cudaFree(d_rloc)); }

double PressureSolver::calculate_pressure(Matrix &P, const Matrix &RS,
                                          const Domain &domain) {
  double res = PressureSolver_kernel(P, RS, domain, omg, d_rloc);

  return res;
};
