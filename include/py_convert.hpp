#ifndef PY_CONVERT_HPP_INCLUDED
#define PY_CONVERT_HPP_INCLUDED

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#include "Eigen/Dense"

auto mat2pyarray(const Eigen::MatrixXd & mat)
    -> pybind11::array_t<double>;

auto pyarray2mat(const pybind11::array_t<double> & array)
    -> Eigen::Map<Eigen::MatrixXd, 0,
                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

#endif // PY_CONVERT_HPP_INCLUDED