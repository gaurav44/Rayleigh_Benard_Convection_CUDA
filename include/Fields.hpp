#pragma once 
#include "DataStructure.hpp"

class Fields {
    public: 
        Fields() = default;

        Matrix U;
        Matrix V;
        Matrix F;
        Matrix G;
        Matrix T;
        Matrix T_old;
        Matrix P;
        Matrix RS;

};
