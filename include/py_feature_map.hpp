#ifndef PY_FEATURE_MAP_HPP_INCLUDED
#define PY_FEATURE_MAP_HPP_INCLUDED

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define WRAP_PYTHON
#include "feature_map.hpp"
#include "py_convert.hpp"

auto py_DecomposableGaussian(pybind11::array_t<double> & A,
                             double sigma)
    -> FeatureMap;

auto py_init_feature_map(pybind11::module & m)
    -> void;

#endif // PY_FEATURE_MAP_HPP_INCLUDED