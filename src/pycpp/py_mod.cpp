#include "py_mod.hpp"

namespace py = pybind11;

PYBIND11_PLUGIN(_pyov2sgd)
{

    py::module m("_pyov2sgd", "Doubly stochastic gradient descent for OVKs");

    py_init_convert(m);
    py_init_loss(m);
    py_init_learning_rate(m);
    py_init_feature_map(m);
    py_init_opt(m);
    // py_init_aopt(m);
    // py_init_bopt(m);
    py_init_IO(m);

    return m.ptr();
}
