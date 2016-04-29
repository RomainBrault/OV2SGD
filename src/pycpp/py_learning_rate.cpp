#include "py_learning_rate.hpp"

namespace py = pybind11;

auto py_init_learning_rate(py::module & m)
    -> void
{
    m.def("InverseScaling",
          &InverseScaling,
          py::arg("gamma0"), py::arg("gamma1"));
    py::class_<LearningRate>(m, "LearningRate")
        .def(py::init<const LearningRate::learning_rate_t &>());
}