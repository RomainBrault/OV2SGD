#ifndef PY_LEARNING_RATE_HPP_INCLUDED
#define PY_LEARNING_RATE_HPP_INCLUDED

#include <pybind11/pybind11.h>

#define WRAP_PYTHON
#include "learning_rate.hpp"

auto py_init_learning_rate(pybind11::module & m)
    -> void;

#endif // PY_LEARNING_RATE_HPP_INCLUDED