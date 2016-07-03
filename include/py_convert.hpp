#ifndef PY_CONVERT_HPP_INCLUDED
#define PY_CONVERT_HPP_INCLUDED

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#ifdef RELEASE
#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#endif
#include "Eigen/Dense"
#include "Eigen/Sparse"

auto mat2pyarray(const Eigen::MatrixXd & mat)
    -> pybind11::array_t<double>;

auto pyarray2mat(const pybind11::array_t<double> & array)
    -> Eigen::Map<Eigen::MatrixXd, 0,
                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

auto py_init_convert(pybind11::module & m)
    -> void;

#endif // PY_CONVERT_HPP_INCLUDED