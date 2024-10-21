#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_reference.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

/**
 * @brief General 2D data structure around std::vector, in column
 * major format.
 *
 */
struct DEV {};
// template <typename T>
class Matrix {
public:
  Matrix(); // = default;

  /**
   * @brief Constructor with initial value
   *
   * @param[in] number of elements in x direction
   * @param[in] number of elements in y direction
   * @param[in] initial value for the elements
   *
   */
  Matrix(int i_max, int j_max, double init_val);

  /**
   * @brief Constructor without an initial value.
   *
   * @param[in] number of elements in x direction
   * @param[in] number of elements in y direction
   *
   */
  Matrix(int i_max, int j_max);

  // Copy constructor
  Matrix(const Matrix &other);
  //     : _imax(other._imax), _jmax(other._jmax), _container(other._container)
  //       /*h_container(other.h_container)*/,
  //       d_container(other._container) {
  //         std::cout << "Copy constructor called\n";
  // }

  Matrix &operator=(const Matrix &other);

  /**
   * @brief Element access and modify using index
   *
   * @param[in] x index
   * @param[in] y index
   * @param[out] reference to the value
   */
  double &operator()(int i, int j);

  thrust::device_reference<double> operator()(DEV, int i, int j);

  /**
   * @brief Element access using index
   *
   * @param[in] x index
   * @param[in] y index
   * @param[out] value of the element
   */
  double operator()(int i, int j) const;

  double operator()(DEV, int i, int j) const;

  void printField(int timestep) const {
    // this->copyToHost();
    std::string fileName = "T" + std::to_string(timestep) + ".txt";
    std::ofstream tmp(fileName);
    for (int j = 1; j < _jmax - 1; j++) {
      for (int i = 1; i < _imax - 1; i++) {
        tmp << std::setprecision(8) << this->operator()(i, j) << ",";
      }
      tmp << "\n";
    }
  }
  
  void printField() const {
    // this->copyToHost();
    std::string fileName = "P.txt";
    std::ofstream tmp(fileName);
    for (int j = 1; j < _jmax - 1; j++) {
      for (int i = 1; i < _imax - 1; i++) {
        tmp << std::setprecision(8) << i << " " << j << " " << this->operator()(i, j) << "\n";
      }
      tmp << "\n";
    }
  }

  void copyToDevice();
  void copyToDevice() const;

  void copyToHost() ;

  //  thrust::host_vector<double> h_container;
  thrust::device_vector<double> d_container;

private:
  // Number of elements in x direction
  int _imax;

  // Number of elements in y direction
  int _jmax;

  // Data container
  std::vector<double> _container;
};

// template class Matrix<float>;
// template class Matrix<double>;
