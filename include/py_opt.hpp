#ifndef PY_OPT_HPP_INCLUDED
#define PY_OPT_HPP_INCLUDED

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define WRAP_PYTHON
#include "opt.hpp"
#include "py_convert.hpp"

auto py_init_opt(pybind11::module & m)
    -> void;

#endif // PY_OPT_HPP_INCLUDED