#include "py_loss.hpp"

namespace py = pybind11;

auto py_init_loss(py::module & m)
    -> void
{
    m.def("SVRLoss", &SVRLoss, py::arg("margin") = 1.);
    m.def("MultitaskSVRLoss", &MultitaskSVRLoss, py::arg("margin") = 1.);
    m.def("RidgeLoss", &RidgeLoss);
    m.def("MultitaskRidgeLoss", &MultitaskRidgeLoss);
    m.def("HingeLoss", &HingeLoss, py::arg("margin") = 1.);
    m.def("SoftMaxLoss", &SoftMaxLoss);
    py::class_<Loss>(m, "Loss")
        .def(py::init<const Loss::loss_dense_t &,
                      const Loss::loss_sparse_t &>())
    ;
}