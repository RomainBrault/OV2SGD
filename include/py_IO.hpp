#ifndef PY_IO_HPP_INCLUDED
#define PY_IO_HPP_INCLUDED

#include <pybind11/pybind11.h>

#define WRAP_PYTHON
#include "IO.hpp"

auto py_init_IO(pybind11::module & m)
    -> void;

#endif // PY_IO_HPP_INCLUDED