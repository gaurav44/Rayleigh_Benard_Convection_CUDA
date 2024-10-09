#include <iostream>
#include "Domain.hpp"
#include "Fields.hpp"
#include "Boundary.hpp"
#include "Simulation.hpp"
#include "PressureSolver.hpp"
#include "DataStructure.hpp"

#define BLOCK_SIZE = 8;

int main() {
    Domain domain;

    domain.xlength = 8.5;
    domain.ylength = 1;
    domain.nu = 0.0296;                      //Kinematic Viscosity
    domain.Re = 1/domain.nu;                 //Reynold's number 
    domain.alpha = 0.00000237;               //Thermal diffusivity
    domain.Pr = domain.nu/domain.alpha;                        //Prandtl number
    domain.beta = 0.00179;                     //Coefficient of thermal expansion (used in Boussinesq approx)
    domain.tau = 0.5;                          //Safety factor for timestep
    domain.gamma = 0.5;                        //Donor-cell scheme factor (will be used in convection)
    domain.GX = 0;                             
    domain.GY = -9.81;                       //Gravitational acceleration
    domain.imax = 85;                         //grid points in x
    domain.jmax = 18;                          //grid points in y
    domain.dx = domain.xlength/domain.imax;   
    domain.dy = domain.ylength/domain.jmax;
    domain.dt = 0.05;                          //Timestep 

    Fields fields;
    fields.U = Matrix<double>(domain.imax+2,domain.jmax+2);
    fields.V = Matrix<double>(domain.imax+2,domain.jmax+2);
    fields.F = Matrix<double>(domain.imax+2,domain.jmax+2);
    fields.G = Matrix<double>(domain.imax+2,domain.jmax+2);
    fields.P = Matrix<double>(domain.imax+2,domain.jmax+2);
    fields.T = Matrix<double>(domain.imax+2,domain.jmax+2, 293.0);
    fields.T_old = Matrix<double>(domain.imax+2,domain.jmax+2, 293.0);
    fields.RS = Matrix<double>(domain.imax+2,domain.jmax+2,0.0); 

    double t_end = 15000;
    double omg = 1.7;                                 //SOR relaxation factor
    double eps = 0.00001;                             //Tolerance for SOR

    int itermax = 500;                             //Maximum iterations for SOR

    double Th = 294.78;                                    
    double Tc = 291.20;

    // Apply Boundaries
    Boundary::apply_boundaries(fields, domain, Th, Tc);
    Boundary::apply_pressure(fields.P, domain);
    double t = 0;
    int timestep = 0;
    // Time loop
    while (t<t_end) {
        Simulation::calculate_dt(domain, fields);
//        fields.U.copyToDevice();
        Simulation::calculate_temperature(fields.U, fields.V, fields.T, domain);
    //     // fields.T.printField();
        Simulation::calculate_fluxes(fields.U, fields.V, fields.T, fields.F, fields.G, domain);
    //     // std::cout << "\n";
    //     // fields.F.printField();
    //     // std::cout << "\n";
    //     // fields.G.printField();
        Simulation::calculate_rs(fields.F, fields.G, fields.RS, domain);
    //     // std::cout << "\n";
    //     // fields.RS.printField();

        int iter = 0;
        double res = 10;

        while (res > eps) {
            if (iter >= itermax) {
                std::cout << "Pressure solver not converged\n";
                std::cout << "dt: " << domain.dt << "Time: " <<  " residual:" << res << " iterations: " << iter << "\n";
                break;
            }
            Boundary::apply_pressure(fields.P, domain);
            // std::cout << "\n";
            // fields.P.printField();
            res = PressureSolver::calculate_pressure(fields.P, fields.RS, domain, omg);
            iter++;
        }
       
        Simulation::calculate_velocities(fields.U, fields.V, fields.F, fields.G, fields.P, domain);
        Boundary::apply_boundaries(fields, domain, Th, Tc);

        if(timestep%500 == 0)
            std::cout << "dt: " << domain.dt << "Time: " << t << " residual:" << res << " iterations: " << iter << "\n";
        t = t + domain.dt;    
        timestep++;
    }
}
