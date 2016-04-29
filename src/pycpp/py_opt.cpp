#include "py_opt.hpp"

namespace py = pybind11;

auto DSOVK::py_fit(const py::array_t<double> & X,
                   const py::array_t<double> & y)
    -> DSOVK &
{
    py::gil_scoped_release release;

    return fit(pyarray2mat(X).transpose(), pyarray2mat(y).transpose());
}

auto DSOVK::py_predict(const py::array_t<double> & X,
                       int nt, long int th)
    -> py::array_t<double>
{
    py::gil_scoped_release release;

    return mat2pyarray(predict(pyarray2mat(X).transpose(), nt, th));
}

auto py_init_opt(py::module & m)
    -> void
{
    py::class_<DSOVK>(m, "DSOVK")
        .def(py::init<const Loss &,
                      const FeatureMap &,
                      const LearningRate &,
                      double, long int,
                      long int, long int, long int, double>())
        .def("fit",
             &DSOVK::py_fit,
             py::arg("X"),
             py::arg("y"))
        .def("predict",
             &DSOVK::py_predict,
             py::arg("X"), py::arg("n_threads") = 4,
             py::arg("thread_threshold") = 10000)
    ;
}
