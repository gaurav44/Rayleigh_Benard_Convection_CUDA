#pragma once

#include "datastructure.hpp"
#include "discretization.hpp"
#include "domain.hpp"
#include "fields.hpp"
#include <cmath>

class Simulation {
public:
  Simulation(Fields* fields, Domain* domain);
  void calculateTimestep();
  void calculateTemperature();

  void calculateFluxes();

  void calculateRightHandSide();

  void calculateVelocities();

private: 
  Fields *_fields;
  Domain *_domain;
};
