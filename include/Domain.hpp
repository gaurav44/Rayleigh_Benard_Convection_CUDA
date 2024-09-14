#pragma once

class Domain {
    public:
        Domain() = default;

        double xlength;
        double ylength;
        double nu;
        double Re;
        double alpha;
        double Pr;
        double beta;
        double tau;
        double gamma;
        double GX;
        double GY;
        int imax;
        int jmax;
        double dx;
        double dy;
        double dt;
};