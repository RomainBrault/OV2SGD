#include "py_convert.hpp"

namespace py = pybind11;

auto mat2pyarray(const Eigen::MatrixXd & mat)
    -> py::array_t<double>
{
    py::gil_scoped_acquire acquire;

    auto array = py::array(
        py::buffer_info(nullptr,
                        sizeof(double),
                        py::format_descriptor<double>::value(),
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

    if (info.format != py::format_descriptor<double>::value()) {
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