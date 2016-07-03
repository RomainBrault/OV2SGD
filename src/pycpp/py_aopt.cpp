// #include "py_aopt.hpp"

// namespace py = pybind11;

// auto ADSOVK::py_fit_denseX_densey(const py::array_t<double> & X,
//                                   const py::array_t<double> & y)
//     -> ADSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(pyarray2mat(X).transpose(), pyarray2mat(y).transpose());
// }

// auto ADSOVK::py_fit_denseX_sparsey(const py::array_t<double> & X,
//                                    const Eigen::SparseMatrix<double> & y)
//     -> ADSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(pyarray2mat(X).transpose(), y);
// }

// auto ADSOVK::py_fit_sparseX_densey(const Eigen::SparseMatrix<double> & X,
//                                    const py::array_t<double> & y)
//     -> ADSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(X, pyarray2mat(y).transpose());
// }

// auto ADSOVK::py_fit_sparseX_sparsey(const Eigen::SparseMatrix<double> & X,
//                                     const Eigen::SparseMatrix<double> & y)
//     -> ADSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(X, y);
// }

// auto ADSOVK::py_predict_denseX(const py::array_t<double> & X,
//                                int nt, long int th)
//     -> py::array_t<double>
// {
//     py::gil_scoped_release release;

//     return mat2pyarray(predict(pyarray2mat(X).transpose(), nt, th));
// }

// auto ADSOVK::py_predict_sparseX(const Eigen::SparseMatrix<double> & X,
//                                 int nt, long int th)
//     -> py::array_t<double>
// {
//     py::gil_scoped_release release;

//     return mat2pyarray(predict(X, nt, th));
// }

// auto py_init_aopt(py::module & m)
//     -> void
// {
//     py::class_<ADSOVK>(m, "ADSOVK")
//         .def(py::init<const Loss &,
//                       const DecomposableFeatureMap &,
//                       const LearningRate &,
//                       const LearningRate &,
//                       double, long int,
//                       long int,
//                       long int, long int, long int>())
//         .def("fit_dense",
//              &ADSOVK::py_fit_denseX_densey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_dense_sparse",
//              &ADSOVK::py_fit_denseX_sparsey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_sparse_dense",
//              &ADSOVK::py_fit_sparseX_densey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_sparse",
//              &ADSOVK::py_fit_sparseX_sparsey,
//              py::arg("X"), py::arg("y"))

//         .def("predict_dense",
//              &ADSOVK::py_predict_denseX,
//              py::arg("X"), py::arg("n_threads") = 8,
//              py::arg("thread_threshold") = 10000)
//         .def("predict_sparse",
//              (py::array_t<double> (ADSOVK::*)
//                 (const Eigen::SparseMatrix<double> &, int, long int))
//              &ADSOVK::py_predict_sparseX,
//              py::arg("X"), py::arg("n_threads") = 8,
//              py::arg("thread_threshold") = 10000)
//     ;
// }
