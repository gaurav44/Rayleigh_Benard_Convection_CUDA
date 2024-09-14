#include "Simulation.hpp"

void Simulation::calculate_temperature(const Matrix<double>& U,
                                  const Matrix<double>& V,
                                  Matrix<double>& T,
                                  const Domain& domain) {
    Matrix<double> T_old = T;

    for (int i = 1; i < domain.imax+1; i++) {
        for(int j = 1; j < domain.jmax+1; j++) {
            T(i, j) =
                T_old(i, j) + domain.dt * (domain.alpha * Discretization::diffusion(T_old, domain, i, j) -
                                            Discretization::convection_T(U, V, T_old, domain, i, j));
        }
    }

}

void Simulation::calculate_fluxes(const Matrix<double>& U,
                                  const Matrix<double>& V,
                                  const Matrix<double>& T,
                                  Matrix<double>& F,
                                  Matrix<double>& G,
                                  const Domain& domain) {
    for(int i = 1; i < domain.imax; i++){
        for(int j = 1; j < domain.jmax+1; j++) {
            F(i,j) = U(i,j) + domain.dt*(domain.nu*Discretization::diffusion(U,domain,i,j) 
                                        - Discretization::convection_u(U,V,domain,i,j)) - (domain.beta*domain.dt/2
                                        *(T(i,j) + T(i+1,j)))*domain.GX;
        }       
    }

    for (int i = 1; i < domain.imax+1; i++) {
        for(int j = 1; j < domain.jmax; j++) {
                G(i,j) = V(i,j) + domain.dt*(domain.nu*Discretization::diffusion(V,domain, i,j) 
                                            - Discretization::convection_v(U,V,domain,i,j)) - (domain.beta*domain.dt/2 
                                            *(T(i,j) + T(i,j+1)))*domain.GY;
        }    
    } 
}

void Simulation::calculate_rs(const Matrix<double>& F, 
                              const Matrix<double>& G,
                              Matrix<double>& RS,
                              const Domain& domain) {
    for (int i = 1; i < domain.imax+1; i++) {
        for(int j = 1; j < domain.jmax+1; j++) {
            double term1 = (F(i, j) - F(i - 1, j)) / domain.dx;
            double term2 = (G(i, j) - G(i, j - 1)) / domain.dy;
            RS(i, j) = (term1 + term2) / domain.dt;
        }
    }
}

void Simulation::calculate_velocities(Matrix<double>& U,
                                      Matrix<double>& V,
                                      const Matrix<double>& F,
                                      const Matrix<double>& G,
                                      const Matrix<double>& P,
                                      const Domain& domain) {
    for (int i = 1; i < domain.imax; i++) {
        for(int j = 1; j < domain.jmax+1; j++) {
            U(i, j) = F(i, j) - domain.dt * (P(i + 1, j) - P(i, j)) / domain.dx;
        }
    }

    for (int i = 1; i < domain.imax+1; i++) {
        for(int j = 1; j < domain.jmax; j++) {
            V(i, j) = G(i, j) - domain.dt * (P(i, j + 1) - P(i, j)) / domain.dy;
        }
    }
}