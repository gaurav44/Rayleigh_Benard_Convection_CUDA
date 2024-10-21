#include "simulation.hpp"
Simulation::Simulation(Fields* fields, Domain* domain)
        : _fields(fields), _domain(domain) {}

void Simulation::calculateTimestep() {
    double CFLu = 0.0;
    double CFLv = 0.0;
    double CFLnu = 0.0;
    double CFLt = 0.0;

    double dx2 = _domain->dx * _domain->dx;
    double dy2 = _domain->dy * _domain->dy;

    double u_max = 0;
    double v_max = 0;

    for (int i = 1; i < _domain->imax + 1; i++) {
      for (int j = 1; j < _domain->jmax + 1; j++) {
        u_max = std::max(u_max, fabs(_fields->U(i, j)));
        v_max = std::max(v_max, fabs(_fields->V(i, j)));
      }
    }

    CFLu = _domain->dx / u_max;
    CFLv = _domain->dy / v_max;

    CFLnu = (0.5 / _domain->nu) * (1.0 / (1.0 / dx2 + 1.0 / dy2));
    _domain->dt = std::min(CFLnu, std::min(CFLu, CFLv));

    CFLt = (0.5 / _domain->alpha) * (1.0 / (1.0 / dx2 + 1.0 / dy2));
    _domain->dt = std::min(_domain->dt, CFLt);

    _domain->dt = _domain->tau * _domain->dt;
}


void Simulation::calculateTemperature() {
    Matrix T_old = _fields->T;

    for (int i = 1; i < _domain->imax+1; i++) {
         for(int j = 1; j < _domain->jmax+1; j++) {
             _fields->T(i, j) =
                 T_old(i, j) + _domain->dt * (_domain->alpha * Discretization::diffusion(T_old, *_domain, i, j) -
                                             Discretization::convection_T(_fields->U, _fields->V, T_old, *_domain, i, j));
         }
     }

}

void Simulation::calculateFluxes() {
    for(int i = 1; i < _domain->imax; i++){
        for(int j = 1; j < _domain->jmax+1; j++) {
            _fields->F(i,j) = _fields->U(i,j) + _domain->dt*(_domain->nu*Discretization::diffusion(_fields->U, *_domain, i, j) 
                                        - Discretization::convection_u(_fields->U, _fields->V, *_domain, i, j)) - (_domain->beta*_domain->dt/2
                                        *(_fields->T(i,j) + _fields->T(i+1,j)))*_domain->GX;
        }       
    }

    for (int i = 1; i < _domain->imax+1; i++) {
        for(int j = 1; j < _domain->jmax; j++) {
                _fields->G(i,j) = _fields->V(i,j) + _domain->dt*(_domain->nu*Discretization::diffusion(_fields->V,*_domain, i,j) 
                                            - Discretization::convection_v(_fields->U,_fields->V,*_domain,i,j)) - (_domain->beta*_domain->dt/2 
                                            *(_fields->T(i,j) + _fields->T(i,j+1)))*_domain->GY;
        }    
    } 
}

void Simulation::calculateRightHandSide() {
    for (int i = 1; i < _domain->imax+1; i++) {
        for(int j = 1; j < _domain->jmax+1; j++) {
            double term1 = (_fields->F(i, j) - _fields->F(i - 1, j)) / _domain->dx;
            double term2 = (_fields->G(i, j) - _fields->G(i, j - 1)) / _domain->dy;
            _fields->RS(i, j) = (term1 + term2) / _domain->dt;
        }
    }
}

void Simulation::calculateVelocities() {
    for (int i = 1; i < _domain->imax; i++) {
        for(int j = 1; j < _domain->jmax+1; j++) {
            _fields->U(i, j) = _fields->F(i, j) - _domain->dt * (_fields->P(i + 1, j) - _fields->P(i, j)) / _domain->dx;
        }
    }

    for (int i = 1; i < _domain->imax+1; i++) {
        for(int j = 1; j < _domain->jmax; j++) {
            _fields->V(i, j) = _fields->G(i, j) - _domain->dt * (_fields->P(i, j + 1) - _fields->P(i, j)) / _domain->dy;
        }
    }
}
