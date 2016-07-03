#include "py_feature_map.hpp"

namespace py = pybind11;

auto py_DivergenceFreeGaussian(double sigma, long int p)
   -> DivergenceFreeFeatureMap
{
    return DivergenceFreeGaussian(sigma, p);
}

auto py_DecomposableGaussian(py::array_t<double> & A, double sigma)
    -> DecomposableFeatureMap
{
    return DecomposableGaussian(pyarray2mat(A).transpose(), sigma);
}

auto py_DecomposableGaussianB(py::array_t<double> & B, double sigma)
    -> DecomposableFeatureMap
{
    return DecomposableGaussianB(pyarray2mat(B).transpose(), sigma);
}

auto py_DecomposableSkewedChi2(py::array_t<double> & A, double skewness)
    -> DecomposableFeatureMap
{
    return DecomposableSkewedChi2(pyarray2mat(A).transpose(), skewness);
}

auto py_DecomposableSkewedChi2B(py::array_t<double> & B, double skewness)
    -> DecomposableFeatureMap
{
    return DecomposableSkewedChi2(pyarray2mat(B).transpose(), skewness);
}

auto py_TransformableGaussian(double sigma)
    -> TransformableFeatureMap
{
    return TransformableGaussian(sigma);
}

auto py_init_feature_map(py::module & m)
    -> void
{
    m.def("DivergenceFreeGaussian",
          &py_DivergenceFreeGaussian,
          py::arg("sigma"), py::arg("p"));
    m.def("DecomposableGaussian",
          &py_DecomposableGaussian,
          py::arg("A"), py::arg("sigma"));
    m.def("DecomposableGaussianB",
          &py_DecomposableGaussianB,
          py::arg("B"), py::arg("sigma"));
    m.def("DecomposableSkewedChi2",
          &py_DecomposableSkewedChi2,
          py::arg("A"), py::arg("skewness"));
    m.def("DecomposableSkewedChi2B",
          &py_DecomposableSkewedChi2,
          py::arg("B"), py::arg("skewness"));
    m.def("TransformableGaussian",
          &py_TransformableGaussian,
          py::arg("sigma"));
    py::class_<DecomposableFeatureMap>(m, "DecomposableFeatureMap")
        .def(py::init<const DecomposableFeatureMap::feature_sampler_t &,
                      const DecomposableFeatureMap::feature_map_dense_t &,
                      const DecomposableFeatureMap::feature_map_sparse_t &,
                      long int, long int>());
    py::class_<TransformableFeatureMap>(m, "TransformableFeatureMap")
        .def(py::init<const TransformableFeatureMap::feature_sampler_t &,
                      const TransformableFeatureMap::feature_map_dense_t &,
                      const TransformableFeatureMap::feature_map_sparse_t &>());
    py::class_<DivergenceFreeFeatureMap>(m, "DivergenceFreeFeatureMap")
        .def(py::init<const DivergenceFreeFeatureMap::feature_sampler_t &,
                      const DivergenceFreeFeatureMap::feature_map_dense_t &,
                      const DivergenceFreeFeatureMap::feature_map_sparse_t &,
                      long int>());
}