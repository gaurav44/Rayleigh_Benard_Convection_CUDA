#include "DataStructure.hpp"

// template<typename T>
Matrix::Matrix() {}

// template <typename T>
Matrix::Matrix(int i_max, int j_max, double init_val)
    : _imax(i_max), _jmax(j_max) {
  _container.resize(i_max * j_max);
  //h_container.resize(i_max * j_max);
  d_container.resize(i_max * j_max);
  std::fill(_container.begin(), _container.end(), init_val);
 // thrust::fill(h_container.begin(), h_container.end(), init_val);
  thrust::fill(d_container.begin(), d_container.end(), init_val);
}

// template <typename T>
Matrix::Matrix(int i_max, int j_max) : _imax(i_max), _jmax(j_max) {
  _container.resize(i_max * j_max);
//  h_container.resize(i_max * j_max);
  d_container.resize(i_max * j_max);
}

Matrix::Matrix(const Matrix &other) : _imax(other._imax), _jmax(other._jmax)
        /*h_container(other.h_container)*/
{
    _container.resize(_imax*_jmax);
    d_container.resize(_imax*_jmax);
    std::copy(other._container.begin(), other._container.end(),_container.begin());
    thrust::copy(other._container.begin(), other._container.end(), d_container.begin());
    // std::cout << "Copy constructor called\n";
}

// template<typename T>
double &Matrix::operator()(int i, int j) { return _container[_imax * j + i]; }

thrust::device_reference<double> Matrix::operator()(DEV, int i, int j) {
  return d_container[_imax * j + i];
}

// template<typename T>
double Matrix::operator()(int i, int j) const {
  return _container[_imax * j + i];
}

double Matrix::operator()(DEV, int i, int j) const {
  return d_container[_imax * j + i];
}

// template<typename T>
Matrix &Matrix::operator=(const Matrix &other) {
  if (this != &other) {
    _imax = other._imax;
    _jmax = other._jmax;
    
    _container.resize(_imax*_jmax);
    d_container.resize(_imax*_jmax);
    std::cout << "copy assignment called\n";
//    h_container = other.h_container;
    thrust::copy(other._container.begin(), other._container.end(), d_container.begin());
    _container = other._container;
  }
  return *this;
}

// template<typename T>
void Matrix::copyToDevice() {
  thrust::copy(_container.begin(), _container.end(), d_container.begin());
}

void Matrix::copyToHost() {
  cudaDeviceSynchronize();
  thrust::copy(d_container.begin(), d_container.end(), _container.begin());
}

// void Matrix::copyToDevice() const {
//   thrust::copy(d_container.begin(), d_container.end(), _container.begin());
// }
//  template class Matrix<float>;
//  template class Matrix<double>;
