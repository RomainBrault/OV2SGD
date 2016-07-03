#ifndef PY_FEATURE_MAP_HPP_INCLUDED
#define PY_FEATURE_MAP_HPP_INCLUDED

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define WRAP_PYTHON
#include "feature_map.hpp"
#include "py_convert.hpp"

auto py_DecomposableGaussian(pybind11::array_t<double> & A,
                             double sigma)
    -> DecomposableFeatureMap;

auto py_DecomposableGaussianB(pybind11::array_t<double> & A,
                              double sigma)
    -> DecomposableFeatureMap;

auto py_DecomposableSkewedChi2(pybind11::array_t<double> & A,
                               double skewness)
    -> DecomposableFeatureMap;

auto py_DecomposableSkewedChi2B(pybind11::array_t<double> & B,
                                double skewness)
    -> DecomposableFeatureMap;

auto py_TransformableGaussian(double sigma)
    -> TransformableFeatureMap;

auto py_init_feature_map(pybind11::module & m)
    -> void;

#endif // PY_FEATURE_MAP_HPP_INCLUDED