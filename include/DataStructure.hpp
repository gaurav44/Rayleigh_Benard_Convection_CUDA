#pragma once

#include <iostream>
#include <vector>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>

/**
 * @brief General 2D data structure around std::vector, in column
 * major format.
 *
 */
//template <typename T>
class Matrix {
 public:
  Matrix();// = default;

  /**
   * @brief Constructor with initial value
   *
   * @param[in] number of elements in x direction
   * @param[in] number of elements in y direction
   * @param[in] initial value for the elements
   *
   */
  // Matrix(int i_max, int j_max, double init_val)
  //     : _imax(i_max), _jmax(j_max) {
  //   _container.resize(i_max * j_max);
  //   std::fill(_container.begin(), _container.end(), init_val);
  // }
  Matrix(int i_max, int j_max, double init_val);
  //     : _imax(i_max), _jmax(j_max) {
  //   _container.resize(i_max * j_max);
  //   std::fill(_container.begin(), _container.end(), init_val);
  // }

  /**
   * @brief Constructor without an initial value.
   *
   * @param[in] number of elements in x direction
   * @param[in] number of elements in y direction
   *
   */
  Matrix(int i_max, int j_max);
  //  : _imax(i_max), _jmax(j_max) {
  //   _container.resize(i_max * j_max);
  // }

  // /**
  //  * @brief Element access and modify using index
  //  *
  //  * @param[in] x index
  //  * @param[in] y index
  //  * @param[out] reference to the value
  //  */
  double &operator()(int i, int j); //{ return _container.at(_imax * j + i); }

  // /**
  //  * @brief Element access using index
  //  *
  //  * @param[in] x index
  //  * @param[in] y index
  //  * @param[out] value of the element
  //  */
  double operator()(int i, int j) const;// { return _container.at(_imax * j + i); }

  // /// get the number of elements in x direction
  // int imax() const { return _imax; }

  // /// get the number of elements in y direction
  // int jmax() const { return _jmax; }

  // void printField() {
  //   for(int j = 0; j < _jmax; j++) {
  //     for(int i = 0; i < _imax; i++) {
  //       std::cout << this->operator()(i,j) << " ";
  //     }
  //     std::cout << "\n";
  //   }
  // }


 private:
  /// Number of elements in x direction
  int _imax;
  /// Number of elements in y direction
  int _jmax;

  /// Data container
  std::vector<double> _container;
  
};

// template class Matrix<float>;
// template class Matrix<double>;
