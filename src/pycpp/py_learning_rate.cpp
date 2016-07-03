#include "py_learning_rate.hpp"

namespace py = pybind11;

auto py_init_learning_rate(py::module & m)
    -> void
{
    m.def("InverseScaling",
          &InverseScaling,
          py::arg("gamma0") = 1, py::arg("gamma1") = 1, py::arg("alpha") = -1);
    m.def("AverageScaling",
          &AverageScaling,
           py::arg("d") = 0, py::arg("n") = 0);
    py::class_<LearningRate>(m, "LearningRate")
        .def(py::init<const LearningRate::learning_rate_t &>());
}