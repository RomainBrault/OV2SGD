// #include "py_bopt.hpp"

// namespace py = pybind11;

// auto BDSOVK::py_fit_denseX_densey(const py::array_t<double> & X,
//                                  const py::array_t<double> & y)
//     -> BDSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(pyarray2mat(X).transpose(), pyarray2mat(y).transpose());
// }

// auto BDSOVK::py_fit_denseX_sparsey(const py::array_t<double> & X,
//                                   const Eigen::SparseMatrix<double> & y)
//     -> BDSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(pyarray2mat(X).transpose(), y);
// }

// auto BDSOVK::py_fit_sparseX_densey(const Eigen::SparseMatrix<double> & X,
//                                   const py::array_t<double> & y)
//     -> BDSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(X, pyarray2mat(y).transpose());
// }

// auto BDSOVK::py_fit_sparseX_sparsey(const Eigen::SparseMatrix<double> & X,
//                                    const Eigen::SparseMatrix<double> & y)
//     -> BDSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(X, y);
// }

// auto BDSOVK::py_predict_denseX(const py::array_t<double> & X,
//                               int nt, long int th)
//     -> py::array_t<double>
// {
//     py::gil_scoped_release release;

//     return mat2pyarray(predict(pyarray2mat(X).transpose(), nt, th));
// }

// auto BDSOVK::py_predict_sparseX(const Eigen::SparseMatrix<double> & X,
//                                int nt, long int th)
//     -> py::array_t<double>
// {
//     py::gil_scoped_release release;

//     return mat2pyarray(predict(X, nt, th));
// }

// auto BDSOVK::py_get_coefs(void) const
//     -> py::array_t<double>
// {
//     return mat2pyarray(get_coefs());
// }

// auto BDSOVK::py_get_B(void) const
//     -> py::array_t<double>
// {
//     return mat2pyarray(get_B());
// }

// auto py_init_bopt(py::module & m)
//     -> void
// {
//     py::class_<BDSOVK>(m, "BDSOVK")
//         .def(py::init<const Loss &,
//                       const FeatureMap &,
//                       const LearningRate &,
//                       double, double, long int,
//                       long int,
//                       long int, long int, long int, double>())
//         .def("fit_dense",
//              &BDSOVK::py_fit_denseX_densey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_dense_sparse",
//              &BDSOVK::py_fit_denseX_sparsey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_sparse_dense",
//              &BDSOVK::py_fit_sparseX_densey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_sparse",
//              &BDSOVK::py_fit_sparseX_sparsey,
//              py::arg("X"), py::arg("y"))

//         .def("predict_dense",
//              &BDSOVK::py_predict_denseX,
//              py::arg("X"), py::arg("n_threads") = 8,
//              py::arg("thread_threshold") = 10000)
//         .def("predict_sparse",
//              (py::array_t<double> (BDSOVK::*)
//                 (const Eigen::SparseMatrix<double> &, int, long int))
//              &BDSOVK::py_predict_sparseX,
//              py::arg("X"), py::arg("n_threads") = 8,
//              py::arg("thread_threshold") = 10000)

//         .def("coefs", &BDSOVK::py_get_coefs)
//         .def("B", &BDSOVK::py_get_B)
//     ;
// }
