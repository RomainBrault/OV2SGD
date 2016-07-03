#include "py_IO.hpp"

namespace py = pybind11;

auto py_init_IO(py::module & m)
    -> void
{
    m.def("read_sparse_char2double",
          &binary::read_sparse<uint8_t, double>,
          py::arg("filename"));
}