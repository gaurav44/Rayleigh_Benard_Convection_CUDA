#pragma once
#include <fstream>
#include <sstream>
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
    
    void readDomainParameters(const std::string &filename) {
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line)) {
      std::stringstream iss(line);
      std::string key;
      double value;
      if (iss >> key >> value) {
        if (key == "xlength")
          xlength = value;
        else if (key == "ylength")
          ylength = value;
        else if (key == "nu")
          nu = value;
        else if (key == "alpha")
          alpha = value;
        else if (key == "beta")
          beta = value;
        else if (key == "tau")
          tau = value;
        else if (key == "gamma")
          gamma = value;
        else if (key == "GX")
          GX = value;
        else if (key == "GY")
          GY = value;
        else if (key == "imax")
          imax = static_cast<int>(value);
        else if (key == "jmax")
          jmax = static_cast<int>(value);
        else if (key == "dt")
          dt = value;
      }
    }
    // Calculate derived quantities
    Re = 1.0 / nu;
    Pr = nu / alpha;
    dx = xlength / imax;
    dy = ylength / jmax;
  }
};