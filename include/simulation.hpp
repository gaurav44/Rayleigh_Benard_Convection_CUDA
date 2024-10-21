#pragma once

#include "datastructure.hpp"
#include "discretization.hpp"
#include "domain.hpp"
#include "fields.hpp"
#include "cuda_utils.hpp"
#include <cmath>
#include <thrust/device_vector.h>

#include "temperature_kernels.hpp"
#include "fluxes_kernels.hpp"
#include "right_hand_side_kernels.hpp"
#include "velocity_kernels.hpp"
#include "timestep_kernels.hpp"

class Simulation {
public:
  Simulation(Fields *fields, Domain *domain);
  ~Simulation();
  void calculateTimeStep();

  void calculateTemperature();

  void calculateFluxes();

  void calculateRightHandSide();

  void calculateVelocities();

  Matrix &getTemperature() { return _fields->T; }
  Matrix &getPressure() { return _fields->P; }
  Matrix &getRightHandSide() { return _fields->RS; }

  void copyAllToDevice() {
    _fields->U.copyToDevice();
    _fields->V.copyToDevice();
    _fields->F.copyToDevice();
    _fields->G.copyToDevice();
    _fields->P.copyToDevice();
    _fields->T.copyToDevice();
  }

  // Fields &getFields() { return _fields; }
  Fields *_fields;
  Domain *_domain;
  double *h_uBlockMax;
  double *h_vBlockMax;
  double *d_uBlockMax;
  double *d_vBlockMax;
  
};





