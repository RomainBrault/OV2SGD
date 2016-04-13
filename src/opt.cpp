#define WRAP_PYTHON 1
#include "opt.hpp"

using namespace Eigen;

auto dec_gauss_rff(MatrixXd B)
    -> std::function<void(const Ref<const MatrixXd>,
                          const Ref<const MatrixXd>,
                          Ref<MatrixXd>,
                          Ref<MatrixXd>)>
{
    return [B](const Ref<const MatrixXd> W,
               const Ref<const MatrixXd> X,
               Ref<MatrixXd> phi_w, Ref<MatrixXd> Z) mutable
    // Inputs: W     of size d x D
    //         X     of size d x n
    // Output: phi_w of size 2.r.D x p.n
    // Buffer: Z     of size D x n
    {
        long int n = X.cols();
        long int p = B.rows();
        long int r = B.cols();
        long int D = W.cols();
        Z.block(0, 0, D, n).noalias() = W.transpose() * X;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < D; ++j) {
                phi_w.block((2 * j    ) * r, i * p, r, p).noalias() =
                    B * cos(Z(j, i));
                phi_w.block((2 * j + 1) * r, i * p, r, p).noalias() =
                    B * sin(Z(j, i));
            }
        }
        phi_w /= sqrt(n);
    };
}

auto DecomposableGaussian(const Ref<const MatrixXd> A,
                          double sigma, long int seed)
    -> std::tuple<std::function<double()>,
                  std::function<void(const Ref<const MatrixXd>,
                                     const Ref<const MatrixXd>,
                                     Ref<MatrixXd>,
                                     Ref<MatrixXd>)>,
                  long int>
{

    SelfAdjointEigenSolver<MatrixXd> svd(A);
    return std::make_tuple(
        std::bind(std::normal_distribution<>(0, 1. / sigma),
                  std::mt19937(seed)),
        dec_gauss_rff(svd.eigenvectors() *
                      svd.eigenvalues().array().sqrt().matrix().asDiagonal()),
        svd.eigenvectors().cols()
    );
}


inline auto DSOVK::_condition(Ref<MatrixXd> mat, double lbda)
    -> void
{
    for (auto i = 0; i < mat.rows(); ++i) {
        mat(i, i) += lbda;
    }
}

auto DSOVK::_predict(const Ref<const MatrixXd> X,
                     Ref<MatrixXd> phi_w,
                     Ref<MatrixXd> pred,
                     Ref<MatrixXd> Z)
    -> MatrixXd
// Inputs: X     of size d x n
//         phi_w of size p.n x 2.r.D
// Output: pred  of size p x n initialized to 0
// Buffer: Z     of size D x n
{
    long int p = pred.rows();
    long int n = pred.cols();
    Map<VectorXd> pred_vview(pred.data(), p * n);
    for (long int j = 0; j < _t; ++j) {
        // get block random feature operator
        _phi(_W.middleCols(j * _block, _block), X, phi_w, Z);
        pred_vview.noalias() +=
            phi_w.transpose() *
            _coefs.segment(2 * _r * j * _block, 2 * _r * _block);
    }
    return pred;
}

DSOVK::DSOVK(const loss_type & gloss,
             const feature_map & fm,
             const gamma_type & gamma, double nu, long int T,
             long int batch, long int block, double cond) :
    _gloss(gloss),
    _phi(std::get<1>(fm)),
    _feature_sampler(std::get<0>(fm)),
    _gamma(gamma), _nu(nu),
    _T(T), _t(0),
    _r(std::get<2>(fm)), _d(0), _p(0),
    _batch(batch), _block(block), _cond(cond),
    _coefs(2 * std::get<2>(fm) * T * block), _W()
{

}

auto DSOVK::predict(const Ref<const MatrixXd> X,
                    Ref<MatrixXd> pred,
                    int nt, long int th)
    -> MatrixXd
{
    long int n = X.cols();
    long int n_batch = n / _batch;
    pred.setZero();

#pragma omp parallel num_threads(nt) if(n * _T * _r * _block > th)
{
    MatrixXd phi_w(2 * _r * _block, _p * _batch);
    MatrixXd Z(_block, _batch);

#pragma omp for
    for (long int i = 0; i < n_batch; ++i) {
        long int X_batch_b = (i * _batch) % n;
        long int X_batch_e = std::min(X_batch_b + _batch, n);
        long int c_batch = X_batch_e - X_batch_b;
        _predict(X.middleCols(X_batch_b, c_batch),
                 phi_w.leftCols(_p * c_batch),
                 pred.middleCols(X_batch_b, c_batch),
                 Z.leftCols(c_batch));
    }
}
    return pred;
}

auto DSOVK::predict(const Ref<const MatrixXd> X)
    -> MatrixXd
{
    MatrixXd pred(_p, X.cols());
    predict(X, pred);
    return pred;
}

auto DSOVK::fit(const Ref<const MatrixXd> X,
                const Ref<const MatrixXd> y)
    -> DSOVK &
    // Precondition: X are shuffled
{
    _d = X.rows();
    _p = y.rows();

    _W.resize(_d, _T * _block);
    for (long int j = 0; j < _T * _block; ++j) {
        for (long int i = 0; i < _d; ++i) {
            _W(i, j) = _feature_sampler();
        }
    }

    MatrixXd pred(_p, _batch);
    MatrixXd residues(_p, _batch);
    MatrixXd preconditioner(2 * _r * _block, 2 * _r * _block);
    VectorXd coef_seg(2 * _r * _block);
    MatrixXd phi_w(2 * _r * _block, _p * _batch);
    MatrixXd Z(_block, _batch);

    LLT<MatrixXd> llt(2 * _r * _block);

    long int n = X.cols();
    _coefs.setZero();
    for (_t = 0; _t < _T; ++_t) {
        long int X_batch_b = (_t * _batch) % n;
        long int X_batch_e = std::min(X_batch_b + _batch, n);
        long int c_batch = X_batch_e - X_batch_b;
        // if (_t % 10 == 0) {
        //     // std::cout << X_batch_b << ' ' << X_batch_e << std::endl;
        //     auto se = (predict(X) - y).squaredNorm();
        //     // std::cout << predict(X) << std::endl;
        //     std::cerr << se / X.cols() << " " << 1 - se / y.squaredNorm() << ' ' << std::endl;
        //     // std::cout << _coefs.transpose() << std::endl;
        //     // std::cout << "NEXT" << std::endl;
        //     // std::cerr << "T: " << _t << std::endl;
        // }
        pred.setZero();
        _predict(X.middleCols(X_batch_b, c_batch), phi_w.leftCols(_p * c_batch),
                 pred.leftCols(c_batch), Z.leftCols(c_batch));
        _gloss(pred.leftCols(c_batch), y.middleCols(X_batch_b, c_batch),
               residues.leftCols(c_batch));

        _phi(_W.middleCols(_t * _block, _block),
             X.middleCols(X_batch_b, c_batch),
             phi_w.leftCols(_p * c_batch), Z.leftCols(c_batch));

        preconditioner.noalias() = phi_w.leftCols(_p * c_batch) *
                                   phi_w.leftCols(_p * c_batch).transpose() /
                                   c_batch;
        _condition(preconditioner, _nu + _cond);
        llt.compute(preconditioner);

        double step_size = _gamma(_t);
        Map<VectorXd> grad_vview(residues.data(), _p * c_batch);

        coef_seg.noalias() =
            (phi_w.leftCols(_p * c_batch) * grad_vview) / c_batch +
            _nu * _coefs.segment(2 * _r * _t * _block, 2 * _r * _block);
        llt.solveInPlace(coef_seg);
        _coefs.segment(2 * _r * _t * _block, 2 * _r * _block).noalias() -=
            step_size * coef_seg;

        _coefs.head(2 * _r * _t * _block) *= (1 - step_size * _nu);
    }
    return *this;
}

#if WRAP_PYTHON

using namespace boost::python;

namespace np = boost::numpy;

ndarray2MatrixXd(np::ndarray & mat)
    -> Map<MatrixXd>
{
    if (mat.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        p::throw_error_already_set();
    }
    if (mat.get_nd() != 2) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        p::throw_error_already_set();
    }
    return Map<MatrixXd>(reinterpret_cast<double*>(mat.get_data()),
                         array.shape(0), array.shape(1))
}

auto dec_gauss_rff_python(const np::ndarray & B)
    -> std::function<void(const Ref<const MatrixXd>,
                          const Ref<const MatrixXd>,
                          Ref<MatrixXd>,
                          Ref<MatrixXd>)>
{
    return dec_gauss_rff(ndarray2MatrixXd(B));
}

// int FooClass::foo_python(PyObject* barIn, PyObject* barOut){
//     Map<VectorXd> _barIn((double *) PyArray_DATA(barIn),m);
//     Map<VectorXd> _barOut((double *) PyArray_DATA(barOut),m);
//     return foo(_barIn, _barOut);
// }

BOOST_PYTHON_MODULE(_ovdsgd)
{

    np::initialize();

    def("dec_gauss_rff", &dec_gauss_rff_python);

    class_<DSOVK>("DSOVK",
        init<DSOVK::loss_type, DSOVK::feature_map, DSOVK::gamma_type,
             double, long int, long int, long int, double>(args("loss",
                                                                "feature_map",
                                                                "gamma",
                                                                "nu",
                                                                "T",
                                                                "batch",
                                                                "block",
                                                                "cond")))
    ;
}
#endif // WRAP_PYTHON
