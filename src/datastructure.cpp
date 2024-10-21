#include "datastructure.hpp"

//template<typename T>
Matrix::Matrix() {}

//template <typename T>
Matrix::Matrix(int i_max, int j_max, double init_val)
      : _imax(i_max), _jmax(j_max) {
    _container.resize(i_max * j_max);
    std::fill(_container.begin(), _container.end(), init_val);
}

//template <typename T>
Matrix::Matrix(int i_max, int j_max)
    : _imax(i_max), _jmax(j_max) {
        _container.resize(i_max * j_max);
}

//template<typename T>
double& Matrix::operator()(int i, int j) { return _container[_imax * j + i];}

//template<typename T>
double Matrix::operator()(int i, int j) const { return _container[_imax * j + i];}

//template class Matrix<float>;
//template class Matrix<double>;
