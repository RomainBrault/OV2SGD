#ifndef PY_LOSS_HPP_INCLUDED
#define PY_LOSS_HPP_INCLUDED

#include <pybind11/pybind11.h>

#define WRAP_PYTHON
#include "loss.hpp"

auto py_init_loss(pybind11::module & m)
    -> void;

#endif // PY_LOSS_HPP_INCLUDED