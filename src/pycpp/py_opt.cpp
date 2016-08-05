#include "py_opt.hpp"

namespace py = pybind11;

auto DSOVK::py_fit_denseX_sparsey(const py::array_t<double> & X,
                                  const Eigen::SparseMatrix<double> & y)
    -> DSOVK &
{
    py::gil_scoped_release release;

    return fit(pyarray2mat(X).transpose(), y);
}

auto DSOVK::py_fit_sparseX_densey(const Eigen::SparseMatrix<double> & X,
                                  const py::array_t<double> & y)
    -> DSOVK &
{
    py::gil_scoped_release release;

    return fit(X, pyarray2mat(y).transpose());
}

auto DSOVK::py_fit_sparseX_sparsey(const Eigen::SparseMatrix<double> & X,
                                   const Eigen::SparseMatrix<double> & y)
    -> DSOVK &
{
    py::gil_scoped_release release;

    return fit(X, y);
}

auto DSOVK::py_predict_denseX(const py::array_t<double> & X,
                              int nt, long int th)
    -> py::array_t<double>
{
    py::gil_scoped_release release;

    return mat2pyarray(predict(pyarray2mat(X).transpose(), nt, th));
}

auto DSOVK::py_predict_sparseX(const Eigen::SparseMatrix<double> & X,
                               int nt, long int th)
    -> py::array_t<double>
{
    py::gil_scoped_release release;

    return mat2pyarray(predict(X, nt, th));
}



// auto TSOVK::py_fit_denseX_densey(const py::array_t<double> & X,
//                                  const py::array_t<double> & y)
//     -> TSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(pyarray2mat(X).transpose(), pyarray2mat(y).transpose());
// }

// auto TSOVK::py_fit_denseX_sparsey(const py::array_t<double> & X,
//                                   const Eigen::SparseMatrix<double> & y)
//     -> TSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(pyarray2mat(X).transpose(), y);
// }

// auto TSOVK::py_fit_sparseX_densey(const Eigen::SparseMatrix<double> & X,
//                                   const py::array_t<double> & y)
//     -> TSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(X, pyarray2mat(y).transpose());
// }

// auto TSOVK::py_fit_sparseX_sparsey(const Eigen::SparseMatrix<double> & X,
//                                    const Eigen::SparseMatrix<double> & y)
//     -> TSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(X, y);
// }

// auto TSOVK::py_predict_denseX(const py::array_t<double> & X,
//                               int nt, long int th)
//     -> py::array_t<double>
// {
//     py::gil_scoped_release release;

//     return mat2pyarray(predict(pyarray2mat(X).transpose(), nt, th));
// }

// auto TSOVK::py_predict_sparseX(const Eigen::SparseMatrix<double> & X,
//                                int nt, long int th)
//     -> py::array_t<double>
// {
//     py::gil_scoped_release release;

//     return mat2pyarray(predict(X, nt, th));
// }

// auto TSOVK::py_get_coefs(void) const
//     -> py::array_t<double>
// {
//     return mat2pyarray(get_coefs());
// }






// auto DivSOVK::py_fit_denseX_densey(const py::array_t<double> & X,
//                                    const py::array_t<double> & y)
//     -> DivSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(pyarray2mat(X).transpose(), pyarray2mat(y).transpose());
// }

// auto DivSOVK::py_fit_denseX_sparsey(const py::array_t<double> & X,
//                                     const Eigen::SparseMatrix<double> & y)
//     -> DivSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(pyarray2mat(X).transpose(), y);
// }

// auto DivSOVK::py_fit_sparseX_densey(const Eigen::SparseMatrix<double> & X,
//                                     const py::array_t<double> & y)
//     -> DivSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(X, pyarray2mat(y).transpose());
// }

// auto DivSOVK::py_fit_sparseX_sparsey(const Eigen::SparseMatrix<double> & X,
//                                      const Eigen::SparseMatrix<double> & y)
//     -> DivSOVK &
// {
//     py::gil_scoped_release release;

//     return fit(X, y);
// }

// auto DivSOVK::py_predict_denseX(const py::array_t<double> & X,
//                                 int nt, long int th)
//     -> py::array_t<double>
// {
//     py::gil_scoped_release release;

//     return mat2pyarray(predict(pyarray2mat(X).transpose(), nt, th));
// }

// auto DivSOVK::py_predict_sparseX(const Eigen::SparseMatrix<double> & X,
//                                  int nt, long int th)
//     -> py::array_t<double>
// {
//     py::gil_scoped_release release;

//     return mat2pyarray(predict(X, nt, th));
// }

// auto DivSOVK::py_get_coefs(void) const
//     -> py::array_t<double>
// {
//     return mat2pyarray(get_coefs());
// }





auto py_init_opt(py::module & m)
    -> void
{
    py::class_<DSOVK>(m, "DSOVK")
        .def(py::init<const Loss &,
                      const DecomposableFeatureMap &,
                      const LearningRate &,
                      long int,
                      double, double, long int,
                      long int, long int, long int, double>())
        .def("fit_dense",
             (DSOVK & (DSOVK::*) (const Eigen::MatrixXd &,
                                  const Eigen::MatrixXd &))
             &DSOVK::fit, py::arg("X"), py::arg("y"))
        .def("fit_dense_sparse",
             &DSOVK::py_fit_denseX_sparsey,
             py::arg("X"), py::arg("y"))
        .def("fit_sparse_dense",
             &DSOVK::py_fit_sparseX_densey,
             py::arg("X"), py::arg("y"))
        .def("fit_sparse",
             &DSOVK::py_fit_sparseX_sparsey,
             py::arg("X"), py::arg("y"))

        .def("predict_dense",
             &DSOVK::py_predict_denseX,
             py::arg("X"), py::arg("n_threads") = 8,
             py::arg("thread_threshold") = 10000)
        .def("predict_sparse",
             (py::array_t<double> (DSOVK::*)
                (const Eigen::SparseMatrix<double> &, int, long int))
             &DSOVK::py_predict_sparseX,
             py::arg("X"), py::arg("n_threads") = 8,
             py::arg("thread_threshold") = 10000)

        .def("coefs",
            (const Eigen::VectorXd & (DSOVK::*) (void) const)
            &DSOVK::get_coefs)
    ;




//     py::class_<TSOVK>(m, "TSOVK")
//         .def(py::init<const Loss &,
//                       const TransformableFeatureMap &,
//                       const LearningRate &,
//                       long int,
//                       double, double, long int,
//                       long int, long int, long int, double>())
//         .def("fit_dense",
//              &TSOVK::py_fit_denseX_densey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_dense_sparse",
//              &TSOVK::py_fit_denseX_sparsey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_sparse_dense",
//              &TSOVK::py_fit_sparseX_densey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_sparse",
//              &TSOVK::py_fit_sparseX_sparsey,
//              py::arg("X"), py::arg("y"))

//         .def("predict_dense",
//              &TSOVK::py_predict_denseX,
//              py::arg("X"), py::arg("n_threads") = 8,
//              py::arg("thread_threshold") = 10000)
//         .def("predict_sparse",
//              (py::array_t<double> (TSOVK::*)
//                 (const Eigen::SparseMatrix<double> &, int, long int))
//              &TSOVK::py_predict_sparseX,
//              py::arg("X"), py::arg("n_threads") = 8,
//              py::arg("thread_threshold") = 10000)

//         .def("coefs", &TSOVK::py_get_coefs)
//     ;




//     py::class_<DivSOVK>(m, "DivSOVK")
//         .def(py::init<const Loss &,
//                       const DivergenceFreeFeatureMap &,
//                       const LearningRate &,
//                       long int,
//                       double, double, long int,
//                       long int, long int, long int, double>())
//         .def("fit_dense",
//              &DivSOVK::py_fit_denseX_densey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_dense_sparse",
//              &DivSOVK::py_fit_denseX_sparsey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_sparse_dense",
//              &DivSOVK::py_fit_sparseX_densey,
//              py::arg("X"), py::arg("y"))
//         .def("fit_sparse",
//              &DivSOVK::py_fit_sparseX_sparsey,
//              py::arg("X"), py::arg("y"))

//         .def("predict_dense",
//              &DivSOVK::py_predict_denseX,
//              py::arg("X"), py::arg("n_threads") = 8,
//              py::arg("thread_threshold") = 10000)
//         .def("predict_sparse",
//              (py::array_t<double> (DivSOVK::*)
//                 (const Eigen::SparseMatrix<double> &, int, long int))
//              &DivSOVK::py_predict_sparseX,
//              py::arg("X"), py::arg("n_threads") = 8,
//              py::arg("thread_threshold") = 10000)

//         .def("coefs", &DivSOVK::py_get_coefs)
//     ;
}
