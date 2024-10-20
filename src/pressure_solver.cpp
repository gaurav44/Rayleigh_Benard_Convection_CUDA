#include "pressure_solver.hpp"

PressureSolver::PressureSolver(Domain *domain) : _domain(domain) {
  dim3 threadsPerBlock(BLOCK_SIZE_SOR, BLOCK_SIZE_SOR);
  dim3 numBlocks(
      (_domain->imax + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (_domain->jmax + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  CHECK(cudaMalloc(&d_rlocBlock, numBlocks.x * numBlocks.y * sizeof(double)));
  h_rlocBlock.resize(numBlocks.x * numBlocks.y);
}

PressureSolver::~PressureSolver() { CHECK(cudaFree(d_rlocBlock)); }

double PressureSolver::calculatePressure(Matrix &P, const Matrix &RS) {
  double res = PressureSolverKernels::calculatePressureKernel(
      P, RS, _domain, omg, d_rlocBlock, h_rlocBlock);

  return res;
};
