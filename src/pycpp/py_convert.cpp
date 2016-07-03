#include <iostream>
#include "py_convert.hpp"

namespace py = pybind11;

auto mat2pyarray(const Eigen::MatrixXd & mat)
    -> py::array_t<double>
{
    py::gil_scoped_acquire acquire;

    auto array = py::array(
        py::buffer_info(nullptr,
                        sizeof(double),
                        py::format_descriptor<double>::value,
                        2,
                        { static_cast<unsigned long>(mat.cols()),
                          static_cast<unsigned long>(mat.rows())},
                        { sizeof(double),
                          sizeof(double) * mat.cols() })
    );
    auto info = array.request();
    double* data = reinterpret_cast<double*>(info.ptr);
    for (unsigned long i = 0; i < mat.rows(); ++i) {
        for (unsigned long j = 0; j < mat.cols(); ++j) {
            *data = mat(i, j);
            ++data;
        }
    }
    return py::array(info);
}

auto pyarray2mat(const py::array_t<double> & array)
    -> Eigen::Map<Eigen::MatrixXd, 0,
                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
{
    py::gil_scoped_acquire acquire;

    py::buffer_info info = const_cast<py::array_t<double> &>(array).request();

    if (info.format != py::format_descriptor<double>::value) {
        throw std::runtime_error("Incompatible format:"
                                 " expected a double array!");
    }
    if (info.ndim != 2) {
        throw std::runtime_error("Incompatible buffer dimension!");
    }
    size_t s1 = info.strides[1] / sizeof(double);
    size_t s2 = info.strides[0] / sizeof(double);
    double* data = reinterpret_cast<double*>(info.ptr);
    Eigen::Map<Eigen::MatrixXd, 0,
               Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        mat(data, info.shape[0], info.shape[1],
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(s1, s2));
    return mat;
}

auto py_init_convert(pybind11::module & m)
    -> void
{
    py::class_<Eigen::SparseMatrix<double>>(m, "EigenSparseMatrix",
                                            py::object())
        .def(py::init<>())
        .def(py::init<typename Eigen::SparseMatrix<double>::Index,
                      typename Eigen::SparseMatrix<double>::Index>())
        .def(py::init<const Eigen::SparseMatrix<double> &>())

        .def_property_readonly("size",
            [](const Eigen::SparseMatrix<double> &m)
            {
                return m.size();
            })
        .def_property_readonly("cols", &Eigen::SparseMatrix<double>::cols)
        .def_property_readonly("rows", &Eigen::SparseMatrix<double>::rows)

        // /* Arithmetic operators (def_cast forcefully casts the result back to a
        //    Type to avoid type issues with Eigen's crazy expression templates) */
        .def_cast(-py::self)
        .def_cast(py::self + py::self)
        .def_cast(py::self - py::self)
        .def_cast(py::self * py::self)
        .def_cast(py::self * float())
        .def_cast(py::self / float())

        /* Arithmetic in-place operators */
        .def(py::self += py::self)
        .def(py::self -= py::self)
        // .def(py::self *= py::self)
        .def(py::self *= float())
        .def(py::self /= float())

        .def("toDense", [](Eigen::SparseMatrix<double> &m)
            {
                return mat2pyarray(m.toDense());
            })

        /* Other transformations */
        .def("transpose", [](Eigen::SparseMatrix<double> &m)
                -> Eigen::SparseMatrix<double>
            {
                return m.transpose();
            })

        .def_property_readonly("shape",
            [](const Eigen::SparseMatrix<double> & m)
            {
                return std::make_tuple(m.rows(), m.cols());
            })
        .def("__getitem__",
            [](const Eigen::SparseMatrix<double> & m,
               const std::pair<int, int> & indices)
            {
                if (std::get<0>(indices) >= m.rows() ||
                    std::get<1>(indices) >= m.cols()) {
                    throw py::index_error();
                }
                return m.coeff(std::get<0>(indices), std::get<1>(indices));
            })
        .def("__len__", [](const Eigen::SparseMatrix<double> & m)
            {
                return m.size();
            })
        .def("__repr__", [](const Eigen::SparseMatrix<double> & m)
                -> std::string
            {
                std::ostringstream buffer;
                buffer << m << std::endl;
                return buffer.str();
            })
    ;
}
