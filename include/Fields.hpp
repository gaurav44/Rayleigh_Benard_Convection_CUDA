#pragma once 
#include "DataStructure.hpp"

class Fields {
    public: 
        Fields() = default;

        Matrix<double> U;
        Matrix<double> V;
        Matrix<double> F;
        Matrix<double> G;
        Matrix<double> T;
        Matrix<double> T_old;
        Matrix<double> P;
        Matrix<double> RS;

};
