#pragma once

#include "DataStructure.hpp"
#include "Discretization.hpp"
#include "Domain.hpp"
#include "Fields.hpp"
#include <cmath>

class Simulation {
public:
  Simulation(Fields* fields, Domain* domain);
  void calculate_dt();

  void calculate_temperature();

  void calculate_fluxes();

  void calculate_rs();

  void calculate_velocities();
  Matrix &getT() { return _fields->T; }
  Matrix &getP() { return _fields->P; }
  Matrix &getRS() { return _fields->RS; }
  void copyAllToDevice(){
    _fields->U.copyToDevice();
    _fields->V.copyToDevice();
    _fields->F.copyToDevice();
    _fields->G.copyToDevice();
    _fields->P.copyToDevice();
    _fields->T.copyToDevice();
  }
  //Fields &getFields() { return _fields; }
  Fields* _fields;
  Domain* _domain;
};
extern void temperature_kernel(const Matrix &U, const Matrix &V, Matrix &T,
                               const Domain &domain);
extern void F_kernel(const Matrix &U, const Matrix &V, const Matrix &T,
                     Matrix &F, const Domain &domain);
extern void G_kernel(const Matrix &U, const Matrix &V, const Matrix &T,
                     Matrix &G, const Domain &domain);
extern void RS_kernel(const Matrix &F, const Matrix &G, Matrix &RS,
                      const Domain &domain);
extern void U_kernel(Matrix &U, const Matrix &F, const Matrix &P,
                     const Domain &domain);
extern void V_kernel(Matrix &V, const Matrix &G, const Matrix &P,
                     const Domain &domain);
extern std::pair<double, double> Dt_kernel(const Matrix &U, const Matrix &V,
                                           const Domain &domain);
