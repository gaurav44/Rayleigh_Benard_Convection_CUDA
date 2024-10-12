#pragma once

#include "DataStructure.hpp"
#include "Discretization.hpp"
#include "Domain.hpp"
#include "Fields.hpp"
#include <cmath>

class Simulation {
public:
  static void calculate_dt(Domain &domain, const Fields &fields);

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
extern void temperature_kernel(const Matrix &U, const Matrix &V, Matrix &T, const Domain &domain);
extern void F_kernel(const Matrix &U, const Matrix &V, const Matrix &T,
                     Matrix &F, const Domain &domain);
extern void G_kernel(const Matrix &U, const Matrix &V, const Matrix &T,
                     Matrix &G, const Domain &domain);
extern void RS_kernel(const Matrix &F, const Matrix &G, Matrix &RS,
                      const Domain &domain);
extern void U_kernel(Matrix& U, const Matrix& F, const Matrix& P, const Domain& domain);
extern void V_kernel(Matrix& V, const Matrix& G, const Matrix& P, const Domain& domain);
extern std::pair<double,double> Dt_kernel(const Matrix& U, const Matrix& V, const Domain& domain);
