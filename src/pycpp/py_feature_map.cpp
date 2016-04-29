#include "py_feature_map.hpp"

namespace py = pybind11;

auto py_DecomposableGaussian(py::array_t<double> & A, double sigma)
    -> FeatureMap
{
    return DecomposableGaussian(pyarray2mat(A), sigma);
}

auto py_init_feature_map(py::module & m)
    -> void
{
    m.def("DecomposableGaussian",
          &py_DecomposableGaussian,
          py::arg("A"), py::arg("sigma"));
    py::class_<FeatureMap>(m, "FeatureMap")
        .def(py::init<const FeatureMap::feature_sampler_t &,
                      const FeatureMap::feature_map_t &,
                      long int, long int>())
    ;
}