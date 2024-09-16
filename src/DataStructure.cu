#include "DataStructure.hpp"

template<typename T>
Matrix<T>::Matrix() {}

template <typename T>
Matrix<T>::Matrix(int i_max, int j_max, double init_val)
      : _imax(i_max), _jmax(j_max) {
    _container.resize(i_max * j_max);
    d_container.resize(i_max * j_max);
    std::fill(_container.begin(), _container.end(), init_val);
    thrust::fill(d_container.begin(), d_container.end(), init_val);
}

template <typename T>
Matrix<T>::Matrix(int i_max, int j_max)
    : _imax(i_max), _jmax(j_max) {
        _container.resize(i_max * j_max);
        h_container.resize(i_max * j_max);
        d_container.resize(i_max * j_max);
}

template<typename T>
T& Matrix<T>::operator()(int i, int j) { return _container[_imax * j + i];}

template<typename T>
T Matrix<T>::operator()(int i, int j) const { return _container[_imax * j + i];}

template<typename T>
void Matrix<T>::copyToDevice() {
    thrust::copy(d_container.begin(), d_container.end(), _container.begin());
}

template class Matrix<float>;
template class Matrix<double>;