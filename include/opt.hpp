#ifndef OPT_HPP_INCLUDED
#define OPT_HPP_INCLUDED

#include <iostream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <chrono>
#include <limits>

#ifdef NDEBUG
#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#endif
#include "Eigen/Dense"
// #include "Eigen/IterativeLinearSolvers"
#include "unsupported/Eigen/IterativeSolvers"

#ifdef WRAP_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#endif // WRAP_PYTHON

#include "loss.hpp"
#include "feature_map.hpp"
#include "learning_rate.hpp"
#include "prng_engine.hpp"

class DSOVK {

public:
    using loss_t = Loss;
    using feature_map_t = DecomposableFeatureMap;
    using gamma_t = LearningRate::learning_rate_t;
    using feature_sampler_t =  DecomposableFeatureMap::feature_sampler_t;

private:
    loss_t        _gloss;
    feature_map_t _phi;

    gamma_t _gamma;
    double  _nu1;
    double  _nu2;

    long int _T;
    long int _T_cap;
    long int _d;
    long int _p;
    long int _batch;
    long int _block;

    double _cond;

    Eigen::VectorXd _coefs;


private:

    template <typename T>
    static inline auto condition_mat(T & mat, double lbda)
        -> void
    {
        for (long int i = 0; i < mat.rows(); ++i) {
            mat(i, i) = mat(i, i) + lbda;
        }
    }

    template <typename M1, typename M2, typename B1, typename B2, typename B3>
    inline auto batch_predict(const M1 & X, M2 && pred,
                       B1 & phi_w, B2 && W, B3 && Z,
                       sitmo::prng_engine & r_engine, long int t)
        -> Eigen::MatrixXd
    // Inputs: X     of size d x n
    // Output: pred  of size p x n initialized to 0
    // Buffer: phi_w of size p.n x 2.r.D
    //         W     of size d x D
    //         Z     of size D x n
    {
        long int p = pred.rows();
        long int n = pred.cols();
        long int r = get_r();
        long int D = W.cols();

        for (long int j = 0; j < t; ++j) {
            feature_map(X, phi_w, W, Z.leftCols(n), r_engine, j);
            pred.noalias() = pred +
                phi_w.leftCols(n).transpose() *
                get_coefs().segment(2 * r * j * _block, 2 * r * _block);
        }
        return pred;
    }

public:
    DSOVK(const loss_t & gloss,
          const feature_map_t & fm,
          const gamma_t & gamma,
          long int p,
          double nu1, double nu2, long int T,
          long int batch = 100, long int block = 100, long int T_cap = -1,
          double cond = 0.1);

private:

    inline auto set_batch(long int val)
        -> DSOVK &
    {
        _batch = val;
        return *this;
    }

    inline auto set_d(long int val)
        -> DSOVK &
    {
        _d = val;
        return *this;
    }

    inline auto get_coefs(void)
        -> Eigen::VectorXd &
    {
        return _coefs;
    }

public:

    inline auto get_d(void) const
        -> long int
    {
        return _d;
    }

    inline auto get_r(void) const
        -> long int
    {
        return get_feature_map().r();
    }

    inline auto get_p(void) const
        -> long int
    {
        return get_feature_map().p();
    }

    inline auto get_T(void) const
        -> long int
    {
        return _T;
    }

    inline auto get_T_cap(void) const
        -> long int
    {
        return _T_cap > 0 ? std::min(_T_cap, get_T()) : get_T();
    }

    inline auto get_nu1(void) const
        -> double
    {
        return _nu1;
    }

    inline auto get_nu2(void) const
        -> double
    {
        return _nu2;
    }

    inline auto get_batch(void) const
        -> long int
    {
        return _batch;
    }

    inline auto get_block(void) const
        -> long int
    {
        return _block;
    }

    inline auto get_feature_map(void) const
        -> const DecomposableFeatureMap &
    {
        return _phi;
    }

    inline auto get_cond(void) const
        -> double
    {
        return _cond;
    }

    inline auto get_coefs(void) const
        -> const Eigen::VectorXd &
    {
        return _coefs;
    }

    template <typename M1, typename M2, typename M3>
    inline auto loss(const M1& pred, const M2& target, M3&& residuals)
        -> void
    {
        _gloss(pred, target, std::forward<M3>(residuals));
    }

    template <typename T, typename U, typename B1, typename B2>
    inline auto feature_map(const T& X, U& phi_w, B1&& W, B2&& Z,
                            sitmo::prng_engine & r_engine, long int seed)
        -> void
    {

        r_engine.seed(seed);
        _phi(X, phi_w, std::forward<B1>(W), std::forward<B2>(Z), r_engine);
    }

    inline auto learning_rate(long int t)
        -> double
    {
        return _gamma(t);
    }

    template <typename M1, typename M2>
    auto predict(const M1 & X, M2 & pred, int nt, long int th)
        -> DSOVK &
    {
        Eigen::initParallel();
        Eigen::setNbThreads(1);

        long int n = X.cols();
        long int batch = n > get_batch() ? get_batch() : n;
        long int block = get_block();
        long int n_batch = n > get_batch() ? n / get_batch() + 1: 1;
        long int r = get_r();
        long int p =  get_p();
        pred.setZero();

    #pragma omp parallel num_threads(nt) if(n * get_T() * r * block > th)
    {
        sitmo::prng_engine r_engine;

        DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>
            phi_w(get_feature_map().init(batch, block, get_p()));
        Eigen::MatrixXd Z(block, batch);
        Eigen::MatrixXd W(get_d(), block);

    #pragma omp for
        for (long int i = 0; i < n_batch; ++i) {
            for (long int j = 0; j < get_T_cap(); ++j) {
                long int X_batch_b = (i * batch) % n;
                long int X_batch_e = std::min(X_batch_b + batch, n);
                long int c_batch = X_batch_e - X_batch_b;
                // get block random feature operator
                feature_map(X.middleCols(X_batch_b, c_batch), phi_w,
                            W, Z.leftCols(c_batch), r_engine, j);
                pred.middleCols(X_batch_b, c_batch) =
                    pred.middleCols(X_batch_b, c_batch) +
                    phi_w.leftCols(c_batch).transpose() *
                    get_coefs().segment(2 * r * j * block, 2 * r * block);
            }
        }
    }
        Eigen::setNbThreads(0);
        return *this;
    }


    template <typename M1>
    auto predict(const M1 & X, int nt = 4, long int th = 10000)
        -> Eigen::MatrixXd
    {
        Eigen::MatrixXd pred(_p, X.cols());
        predict(X, pred, nt, th);
        return pred;
    }

#ifdef WRAP_PYTHON
    auto py_predict_denseX(const pybind11::array_t<double> & X,
                           int nt, long int th)
        -> pybind11::array_t<double>;

    auto py_predict_sparseX(const Eigen::SparseMatrix<double> & X,
                            int nt, long int th)
        -> pybind11::array_t<double>;
#endif

template <typename M1>
auto Jacobian(const M1 & X)
    -> Eigen::MatrixXd
{

}

template <typename M1, typename M2>
auto fit(const M1 & Xt, const M2 & yt)
    -> DSOVK &
{
    auto X = Xt.transpose().eval();
    auto y = yt.transpose().eval();
    set_d(X.rows());
    set_batch(std::min<typename M1::Index>(get_batch(), X.cols()));

    long int r = get_r();
    long int d = get_d();
    long int p = get_p();
    long int D_w = get_block();
    long int D_phi_w = 2 * r * D_w;
    long int m_batch = get_batch();
    long int T = get_T();

    get_coefs().setZero();

    sitmo::prng_engine r_engine;

    Eigen::MatrixXd W(d, D_w);
    Eigen::MatrixXd pred(p, m_batch);
    Eigen::MatrixXd residues(p, m_batch);
    Eigen::VectorXd coef_seg(D_phi_w);
    Eigen::VectorXd update(D_phi_w);
    Eigen::MatrixXd Z(D_w, m_batch);

    DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>
        phi_w(get_feature_map().init(get_batch(), get_block(), get_p()));
    DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>
        cond_operator(2 * D_w, 2 * D_w, r, r);

    Eigen::ConjugateGradient<DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>,
                             Eigen::Lower|Eigen::Upper,
                             Eigen::IdentityPreconditioner> preconditioner;
    preconditioner.setTolerance(get_cond());

    long int n = X.cols();
    long int next_shuffle = 0;

    for (long int t = 0; t < T; ++t) {
        long int X_batch_b = (t * get_batch()) % n;
        long int X_batch_e = std::min(X_batch_b + get_batch(), n);
        long int c_batch = X_batch_e - X_batch_b;
        long int p_c_batch = p * c_batch;
        long int D_phi_w_b = D_phi_w * (t % get_T_cap());

        /* Shuffle at each new epoch */
        long int cur_epoch = (t * get_batch()) / n;
        if (cur_epoch >= next_shuffle) {
            ++next_shuffle;
            shuffle(X, y, r_engine);
        }

        /* Make a new prediction and compute loss */
        pred.setZero();
        batch_predict(X.middleCols(X_batch_b, c_batch), pred.leftCols(c_batch),
                      phi_w, W, Z, r_engine, std::min(t, get_T_cap()));
        loss(pred.leftCols(c_batch), y.middleCols(X_batch_b, c_batch),
             residues.leftCols(c_batch));

        /* Get fresh feature map on batch data */
        Eigen::Map<Eigen::VectorXd> grad_vview(residues.data(), p_c_batch);
        feature_map(X.middleCols(X_batch_b, c_batch), phi_w,
                    W, Z, r_engine, t % get_T_cap());

        /* Compute the update from the gradient and fresh feature map */
        double step_size = learning_rate(t);
        coef_seg = phi_w.leftCols(c_batch) * grad_vview +
            get_nu1() * get_coefs().segment(D_phi_w_b, D_phi_w);
        // std::cout << coef_seg.transpose() << std::endl;

        /* Precondition the gradient update */
        /* This stuff allocate a huge amount of memory.... */
        cond_operator.get_Lhs().noalias() =
            phi_w.get_Lhs().leftCols(c_batch) *
            phi_w.get_Lhs().leftCols(c_batch).transpose() / c_batch;
        cond_operator.get_Rhs().noalias() =
            phi_w.get_Rhs() *
            phi_w.get_Rhs().transpose();
        condition_mat(cond_operator.get_Lhs(), get_nu1());
        preconditioner.compute(cond_operator);
        update = preconditioner.solve(coef_seg);
        // std::cout << update.transpose() << std::endl;

        /* Perform the update */
        long int head_D_phi_w = D_phi_w_b;
        long int tail_D_phi_w = get_coefs().size() - (D_phi_w_b + D_phi_w);
        double shrink = (1 - step_size * get_nu1());
        get_coefs().head(head_D_phi_w) *= shrink;
        get_coefs().segment(head_D_phi_w, D_phi_w) -= step_size * update;
        get_coefs().tail(tail_D_phi_w) *= shrink;

        // std::cout << predict(X.middleCols(X_batch_b, c_batch)).transpose() << std::endl;
        Eigen::MatrixXd yp(y.middleCols(X_batch_b, c_batch));
        // std::cout << yp.transpose() << std::endl;

        for (long int i = 0; i < get_coefs().size() / (2 * r); ++i) {
            double v = get_coefs().segment(2 * i * r, r).norm();
            if (v > get_nu2() * step_size) {
                get_coefs().segment(2 * i * r, r) =
                    get_coefs().segment(2 * i * r, r) -
                    get_coefs().segment(2 * i * r, r) * get_nu2() * step_size / v;
            }
            else {
                get_coefs().segment(2 * i * r, r).fill(0);
            }
        }
    // std::cout << "NEXT" << std::endl << std::endl;
    }

    // std::cout << "END" << std::endl << std::endl;

    return *this;
}

#ifdef WRAP_PYTHON
    auto py_fit_denseX_sparsey(const pybind11::array_t<double> & X,
                               const Eigen::SparseMatrix<double> & y)
        -> DSOVK &;

    auto py_fit_sparseX_densey(const Eigen::SparseMatrix<double> & X,
                               const pybind11::array_t<double> & y)
        -> DSOVK &;

    auto py_fit_sparseX_sparsey(const Eigen::SparseMatrix<double> & X,
                                const Eigen::SparseMatrix<double> & y)
        -> DSOVK &;

#endif

};



















// class TSOVK {

// public:
//     using loss_t = Loss;
//     using feature_map_t = TransformableFeatureMap;
//     using gamma_t = LearningRate::learning_rate_t;
//     using feature_sampler_t =  TransformableFeatureMap::feature_sampler_t;

// private:
//     loss_t        _gloss;
//     feature_map_t _phi;

//     gamma_t _gamma;
//     double  _nu1;
//     double  _nu2;

//     long int _T;
//     long int _T_cap;
//     long int _d;
//     long int _p;
//     long int _batch;
//     long int _block;

//     double _cond;

//     Eigen::VectorXd _coefs;


// private:

//     template <typename T>
//     static inline auto condition_mat(T & mat, double lbda)
//         -> void
//     {
//         for (long int i = 0; i < mat.rows(); ++i) {
//             mat(i, i) = mat(i, i) + lbda;
//         }
//     }

//     template <typename M1, typename M2, typename B1, typename B2, typename B3>
//     inline auto batch_predict(const M1 & X, M2 && pred,
//                        B1 & phi_w, B2 && W, B3 && Z,
//                        sitmo::prng_engine & r_engine, long int t)
//         -> Eigen::MatrixXd
//     // Inputs: X     of size d x n
//     // Output: pred  of size p x n initialized to 0
//     // Buffer: phi_w of size p.n x 2.r.D
//     //         W     of size d x D
//     //         Z     of size D x n
//     {
//         long int p = pred.rows();
//         long int n = pred.cols();
//         long int D = W.cols();

//         for (long int j = 0; j < t; ++j) {
//             feature_map(X, phi_w, W, Z.leftCols(n), r_engine, j);
//             pred.noalias() = pred +
//                 phi_w.leftCols(n * p).transpose() *
//                 get_coefs().segment(2 * j * _block, 2 * _block);
//         }
//         return pred;
//     }

// public:
//     TSOVK(const loss_t & gloss,
//           const feature_map_t & fm,
//           const gamma_t & gamma,
//           long int p,
//           double nu1, double nu2, long int T,
//           long int batch = 100, long int block = 100, long int T_cap = -1,
//           double cond = 0.1);

// private:

//     inline auto set_batch(long int val)
//         -> TSOVK &
//     {
//         _batch = val;
//         return *this;
//     }

//     inline auto set_d(long int val)
//         -> TSOVK &
//     {
//         _d = val;
//         return *this;
//     }

//     inline auto get_coefs(void)
//         -> Eigen::VectorXd &
//     {
//         return _coefs;
//     }

// public:

//     inline auto get_d(void) const
//         -> long int
//     {
//         return _d;
//     }

//     inline auto get_p(void) const
//         -> long int
//     {
//         return _d;
//     }

//     inline auto get_T(void) const
//         -> long int
//     {
//         return _T;
//     }

//     inline auto get_T_cap(void) const
//         -> long int
//     {
//         return _T_cap > 0 ? std::min(_T_cap, get_T()) : get_T();
//     }

//     inline auto get_nu1(void) const
//         -> double
//     {
//         return _nu1;
//     }

//     inline auto get_nu2(void) const
//         -> double
//     {
//         return _nu2;
//     }

//     inline auto get_batch(void) const
//         -> long int
//     {
//         return _batch;
//     }

//     inline auto get_block(void) const
//         -> long int
//     {
//         return _block;
//     }

//     inline auto get_feature_map(void) const
//         -> const TransformableFeatureMap &
//     {
//         return _phi;
//     }

//     inline auto get_cond(void) const
//         -> double
//     {
//         return _cond;
//     }

//     inline auto get_coefs(void) const
//         -> const Eigen::VectorXd &
//     {
//         return _coefs;
//     }

//     template <typename M1, typename M2, typename M3>
//     inline auto loss(const M1& pred, const M2& target, M3&& residuals)
//         -> void
//     {
//         _gloss(pred, target, std::forward<M3>(residuals));
//     }

//     template <typename T, typename U, typename B1, typename B2>
//     inline auto feature_map(const T& X, U& phi_w, B1&& W, B2&& Z,
//                             sitmo::prng_engine & r_engine, long int seed)
//         -> void
//     {

//         r_engine.seed(seed);
//         _phi(X, phi_w, std::forward<B1>(W), std::forward<B2>(Z), r_engine);
//     }

//     inline auto learning_rate(long int t)
//         -> double
//     {
//         return _gamma(t);
//     }

//     template <typename M1, typename M2>
//     auto predict(const M1 & X, M2 & pred, int nt, long int th)
//         -> TSOVK &
//     {
//         Eigen::initParallel();
//         Eigen::setNbThreads(1);

//         long int n = X.cols();
//         long int batch = n > get_batch() ? get_batch() : n;
//         long int block = get_block();
//         long int n_batch = n > get_batch() ? n / get_batch() + 1: 1;
//         long int p =  get_p();
//         pred.setZero();

//     #pragma omp parallel num_threads(nt) if(n * get_T() * block > th)
//     {
//         sitmo::prng_engine r_engine;

//         Eigen::MatrixXd phi_w(get_feature_map().init(batch, block, get_p()));
//         Eigen::MatrixXd Z(block, batch);
//         Eigen::MatrixXd W(1, block);

//     #pragma omp for
//         for (long int i = 0; i < n_batch; ++i) {
//             for (long int j = 0; j < get_T_cap(); ++j) {
//                 long int X_batch_b = (i * batch) % n;
//                 long int X_batch_e = std::min(X_batch_b + batch, n);
//                 long int c_batch = X_batch_e - X_batch_b;
//                 // get block random feature operator
//                 feature_map(X.middleCols(X_batch_b, c_batch), phi_w,
//                             W, Z.leftCols(c_batch), r_engine, j);
//                 pred.middleCols(X_batch_b, c_batch).noalias() =
//                     pred.middleCols(X_batch_b, c_batch) +
//                     phi_w.leftCols(c_batch * p).transpose() *
//                     get_coefs().segment(2 * j * block, 2 * block);
//             }
//         }
//     }
//         Eigen::setNbThreads(0);
//         return *this;
//     }


//     template <typename M1>
//     auto predict(const M1 & X, int nt = 4, long int th = 10000)
//         -> Eigen::MatrixXd
//     {
//         Eigen::MatrixXd pred(_p, X.cols());
//         predict(X, pred, nt, th);
//         return pred;
//     }

// #ifdef WRAP_PYTHON
//     auto py_predict_denseX(const pybind11::array_t<double> & X,
//                            int nt, long int th)
//         -> pybind11::array_t<double>;

//     auto py_predict_sparseX(const Eigen::SparseMatrix<double> & X,
//                             int nt, long int th)
//         -> pybind11::array_t<double>;
// #endif

// template <typename M1>
// auto Jacobian(const M1 & X)
//     -> Eigen::MatrixXd
// {

// }

// template <typename M1, typename M2>
// auto fit(const M1 & X, const M2 & y)
//     -> TSOVK &
// {
//     set_d(X.rows());
//     set_batch(std::min<typename M1::Index>(get_batch(), X.cols()));

//     long int d = get_d();
//     long int p = get_p();
//     long int D_w = get_block();
//     long int D_phi_w = 2 * D_w;
//     long int m_batch = get_batch();
//     long int T = get_T();

//     get_coefs().setZero();

//     sitmo::prng_engine r_engine;

//     Eigen::MatrixXd W(1, D_w);
//     Eigen::MatrixXd pred(p, m_batch);
//     Eigen::MatrixXd residues(p, m_batch);
//     Eigen::VectorXd coef_seg(D_phi_w);
//     Eigen::VectorXd update(D_phi_w);
//     Eigen::MatrixXd Z(D_w, m_batch);

//     Eigen::MatrixXd
//         phi_w(get_feature_map().init(get_batch(), get_block(), get_p()));
//     Eigen::MatrixXd cond_operator(D_phi_w, D_phi_w);

//     Eigen::ConjugateGradient<Eigen::MatrixXd,
//                              Eigen::Lower|Eigen::Upper,
//                              Eigen::IdentityPreconditioner> preconditioner;
//     preconditioner.setTolerance(get_cond());

//     long int n = X.cols();
//     long int next_shuffle = 0;

//     for (long int t = 0; t < T; ++t) {
//         long int X_batch_b = (t * get_batch()) % n;
//         long int X_batch_e = std::min(X_batch_b + get_batch(), n);
//         long int c_batch = X_batch_e - X_batch_b;
//         long int p_c_batch = p * c_batch;
//         long int D_phi_w_b = D_phi_w * (t % get_T_cap());

//         /* Shuffle at each new epoch */
//         long int cur_epoch = (t * get_batch()) / n;
//         if (cur_epoch >= next_shuffle) {
//             ++next_shuffle;
//             shuffle(const_cast<M1 &>(X), const_cast<M2 &>(y), r_engine);
//         }

//         /* Make a new prediction and compute loss */
//         pred.setZero();
//         batch_predict(X.middleCols(X_batch_b, c_batch), pred.leftCols(c_batch),
//                       phi_w, W, Z, r_engine, std::min(t, get_T_cap()));
//         loss(pred.leftCols(c_batch), y.middleCols(X_batch_b, c_batch),
//              residues.leftCols(c_batch));

//         /* Get fresh feature map on batch data */
//         Eigen::Map<Eigen::VectorXd> grad_vview(residues.data(), p_c_batch);
//         feature_map(X.middleCols(X_batch_b, c_batch), phi_w,
//                     W, Z, r_engine, t % get_T_cap());

//         /* Compute the update from the gradient and fresh feature map */
//         double step_size = learning_rate(t);
//         coef_seg = phi_w.leftCols(c_batch * p) * grad_vview +
//             get_nu1() * get_coefs().segment(D_phi_w_b, D_phi_w);

//         /* Precondition the gradient update */
//         /* This stuff allocate a huge amount of memory.... */
//         cond_operator.noalias() =
//             phi_w.leftCols(c_batch * p) *
//             phi_w.leftCols(c_batch * p).transpose() / c_batch;
//         condition_mat(cond_operator, get_nu1());
//         preconditioner.compute(cond_operator);
//         update = preconditioner.solve(coef_seg);

//         /* Perform the update */
//         get_coefs().segment(D_phi_w_b, D_phi_w) =
//             get_coefs().segment(D_phi_w_b, D_phi_w) - step_size * update;

//         long int head_D_phi_w = D_phi_w_b;
//         long int tail_D_phi_w = get_coefs().size() - (D_phi_w_b + D_phi_w);
//         double shrink = (1 - step_size * get_nu1());
//         get_coefs().head(D_phi_w_b) =
//             get_coefs().head(D_phi_w_b) * shrink;
//         get_coefs().tail(tail_D_phi_w) =
//             get_coefs().tail(tail_D_phi_w) * shrink;
//     }

//     return *this;
// }

// #ifdef WRAP_PYTHON
//     auto py_fit_denseX_densey(const pybind11::array_t<double> & X,
//                               const pybind11::array_t<double> & y)
//         -> TSOVK &;

//     auto py_fit_denseX_sparsey(const pybind11::array_t<double> & X,
//                                const Eigen::SparseMatrix<double> & y)
//         -> TSOVK &;

//     auto py_fit_sparseX_densey(const Eigen::SparseMatrix<double> & X,
//                                const pybind11::array_t<double> & y)
//         -> TSOVK &;

//     auto py_fit_sparseX_sparsey(const Eigen::SparseMatrix<double> & X,
//                                 const Eigen::SparseMatrix<double> & y)
//         -> TSOVK &;

//     auto py_get_coefs() const
//         -> pybind11::array_t<double>;
// #endif

// };



















// class DivSOVK {

// public:
//     using loss_t = Loss;
//     using feature_map_t = DivergenceFreeFeatureMap;
//     using gamma_t = LearningRate::learning_rate_t;
//     using feature_sampler_t =  DivergenceFreeFeatureMap::feature_sampler_t;

// private:
//     loss_t        _gloss;
//     feature_map_t _phi;

//     gamma_t _gamma;
//     double  _nu1;
//     double  _nu2;

//     long int _T;
//     long int _T_cap;
//     long int _d;
//     long int _p;
//     long int _batch;
//     long int _block;

//     double _cond;

//     Eigen::VectorXd _coefs;


// private:

//     template <typename T>
//     static inline auto condition_mat(T & mat, double lbda)
//         -> void
//     {
//         for (long int i = 0; i < mat.rows(); ++i) {
//             mat(i, i) = mat(i, i) + lbda;
//         }
//     }

//     template <typename M1, typename M2, typename B1, typename B2, typename B3>
//     inline auto batch_predict(const M1 & X, M2 && pred,
//                        B1 & phi_w, B2 && W, B3 && Z,
//                        sitmo::prng_engine & r_engine, long int t)
//         -> Eigen::MatrixXd
//     // Inputs: X     of size d x n
//     // Output: pred  of size p x n initialized to 0
//     // Buffer: phi_w of size p.n x 2.r.D
//     //         W     of size d x D
//     //         Z     of size D x n
//     {
//         long int p = pred.rows();
//         long int r = p - 1;
//         long int n = pred.cols();
//         long int D = W.cols();

//         for (long int j = 0; j < t; ++j) {
//             feature_map(X, phi_w, W, Z.leftCols(n), r_engine, j);
//             pred.noalias() = pred +
//                 phi_w.leftCols(n * p).transpose() *
//                 get_coefs().segment(2 * r * j * _block, 2 * r * _block);
//         }
//         return pred;
//     }

// public:
//     DivSOVK(const loss_t & gloss,
//             const feature_map_t & fm,
//             const gamma_t & gamma,
//             long int p,
//             double nu1, double nu2, long int T,
//             long int batch = 100, long int block = 100, long int T_cap = -1,
//             double cond = 0.1);

// private:

//     inline auto set_batch(long int val)
//         -> DivSOVK &
//     {
//         _batch = val;
//         return *this;
//     }

//     inline auto set_d(long int val)
//         -> DivSOVK &
//     {
//         _d = val;
//         return *this;
//     }

//     inline auto get_coefs(void)
//         -> Eigen::VectorXd &
//     {
//         return _coefs;
//     }

// public:

//     inline auto get_d(void) const
//         -> long int
//     {
//         return _d;
//     }

//     inline auto get_p(void) const
//         -> long int
//     {
//         return _d;
//     }

//     inline auto get_r(void) const
//         -> long int
//     {
//         return get_feature_map().r();
//     }

//     inline auto get_T(void) const
//         -> long int
//     {
//         return _T;
//     }

//     inline auto get_T_cap(void) const
//         -> long int
//     {
//         return _T_cap > 0 ? std::min(_T_cap, get_T()) : get_T();
//     }

//     inline auto get_nu1(void) const
//         -> double
//     {
//         return _nu1;
//     }

//     inline auto get_nu2(void) const
//         -> double
//     {
//         return _nu2;
//     }

//     inline auto get_batch(void) const
//         -> long int
//     {
//         return _batch;
//     }

//     inline auto get_block(void) const
//         -> long int
//     {
//         return _block;
//     }

//     inline auto get_feature_map(void) const
//         -> const DivergenceFreeFeatureMap &
//     {
//         return _phi;
//     }

//     inline auto get_cond(void) const
//         -> double
//     {
//         return _cond;
//     }

//     inline auto get_coefs(void) const
//         -> const Eigen::VectorXd &
//     {
//         return _coefs;
//     }

//     template <typename M1, typename M2, typename M3>
//     inline auto loss(const M1& pred, const M2& target, M3&& residuals)
//         -> void
//     {
//         _gloss(pred, target, std::forward<M3>(residuals));
//     }

//     template <typename T, typename U, typename B1, typename B2>
//     inline auto feature_map(const T& X, U& phi_w, B1&& W, B2&& Z,
//                             sitmo::prng_engine & r_engine, long int seed)
//         -> void
//     {

//         r_engine.seed(seed);
//         _phi(X, phi_w, std::forward<B1>(W), std::forward<B2>(Z), r_engine);
//     }

//     inline auto learning_rate(long int t)
//         -> double
//     {
//         return _gamma(t);
//     }

//     template <typename M1, typename M2>
//     auto predict(const M1 & X, M2 & pred, int nt, long int th)
//         -> DivSOVK &
//     {
//         Eigen::initParallel();
//         Eigen::setNbThreads(1);

//         long int n = X.cols();
//         long int batch = n > get_batch() ? get_batch() : n;
//         long int block = get_block();
//         long int n_batch = n > get_batch() ? n / get_batch() + 1: 1;
//         long int p = get_p();
//         long int r = p - 1;
//         pred.setZero();

//     #pragma omp parallel num_threads(nt) if(n * r * get_T() * block > th)
//     {
//         sitmo::prng_engine r_engine;

//         Eigen::MatrixXd phi_w(get_feature_map().init(batch, block, get_p()));
//         Eigen::MatrixXd Z(block, batch);
//         Eigen::MatrixXd W(get_d(), block);

//     #pragma omp for
//         for (long int i = 0; i < n_batch; ++i) {
//             for (long int j = 0; j < get_T_cap(); ++j) {
//                 long int X_batch_b = (i * batch) % n;
//                 long int X_batch_e = std::min(X_batch_b + batch, n);
//                 long int c_batch = X_batch_e - X_batch_b;
//                 // get block random feature operator
//                 feature_map(X.middleCols(X_batch_b, c_batch), phi_w,
//                             W, Z.leftCols(c_batch), r_engine, j);
//                 pred.middleCols(X_batch_b, c_batch).noalias() =
//                     pred.middleCols(X_batch_b, c_batch) +
//                     phi_w.leftCols(c_batch * p).transpose() *
//                     get_coefs().segment(2 * r * j * block, 2 * r * block);
//             }
//         }
//     }
//         Eigen::setNbThreads(0);
//         return *this;
//     }


//     template <typename M1>
//     auto predict(const M1 & X, int nt = 4, long int th = 10000)
//         -> Eigen::MatrixXd
//     {
//         Eigen::MatrixXd pred(_p, X.cols());
//         predict(X, pred, nt, th);
//         return pred;
//     }

// #ifdef WRAP_PYTHON
//     auto py_predict_denseX(const pybind11::array_t<double> & X,
//                            int nt, long int th)
//         -> pybind11::array_t<double>;

//     auto py_predict_sparseX(const Eigen::SparseMatrix<double> & X,
//                             int nt, long int th)
//         -> pybind11::array_t<double>;
// #endif

// template <typename M1>
// auto Jacobian(const M1 & X)
//     -> Eigen::MatrixXd
// {

// }

// template <typename M1, typename M2>
// auto fit(const M1 & X, const M2 & y)
//     -> DivSOVK &
// {
//     set_d(X.rows());
//     set_batch(std::min<typename M1::Index>(get_batch(), X.cols()));

//     long int d = get_d();
//     long int p = get_p();
//     long int r = p - 1;
//     long int D_w = get_block();
//     long int D_phi_w = 2 * r * D_w;
//     long int m_batch = get_batch();
//     long int T = get_T();

//     get_coefs().setZero();

//     sitmo::prng_engine r_engine;

//     Eigen::MatrixXd W(d, D_w);
//     Eigen::MatrixXd pred(p, m_batch);
//     Eigen::MatrixXd residues(p, m_batch);
//     Eigen::VectorXd coef_seg(D_phi_w);
//     Eigen::VectorXd update(D_phi_w);
//     Eigen::MatrixXd Z(D_w, m_batch);

//     Eigen::MatrixXd
//         phi_w(get_feature_map().init(get_batch(), get_block(), get_p()));
//     Eigen::MatrixXd cond_operator(D_phi_w, D_phi_w);

//     Eigen::ConjugateGradient<Eigen::MatrixXd,
//                              Eigen::Lower|Eigen::Upper> preconditioner;
//     preconditioner.setTolerance(get_cond());

//     long int n = X.cols();
//     long int next_shuffle = 0;

//     for (long int t = 0; t < T; ++t) {
//         long int X_batch_b = (t * get_batch()) % n;
//         long int X_batch_e = std::min(X_batch_b + get_batch(), n);
//         long int c_batch = X_batch_e - X_batch_b;
//         long int p_c_batch = p * c_batch;
//         long int D_phi_w_b = D_phi_w * (t % get_T_cap());

//         /* Shuffle at each new epoch */
//         long int cur_epoch = (t * get_batch()) / n;
//         if (cur_epoch >= next_shuffle) {
//             ++next_shuffle;
//             shuffle(const_cast<M1 &>(X), const_cast<M2 &>(y), r_engine);
//         }

//         /* Make a new prediction and compute loss */
//         pred.setZero();
//         batch_predict(X.middleCols(X_batch_b, c_batch), pred.leftCols(c_batch),
//                       phi_w, W, Z, r_engine, std::min(t, get_T_cap()));
//         loss(pred.leftCols(c_batch), y.middleCols(X_batch_b, c_batch),
//              residues.leftCols(c_batch));

//         /* Get fresh feature map on batch data */
//         Eigen::Map<Eigen::VectorXd> grad_vview(residues.data(), p_c_batch);
//         feature_map(X.middleCols(X_batch_b, c_batch), phi_w,
//                     W, Z, r_engine, t % get_T_cap());

//         /* Compute the update from the gradient and fresh feature map */
//         double step_size = learning_rate(t);
//         coef_seg = phi_w.leftCols(c_batch * p) * grad_vview +
//             get_nu1() * get_coefs().segment(D_phi_w_b, D_phi_w);

//         /* Precondition the gradient update */
//         /* This stuff allocate a huge amount of memory.... */
//         cond_operator.noalias() =
//             phi_w.leftCols(c_batch * p) *
//             phi_w.leftCols(c_batch * p).transpose() / c_batch;
//         condition_mat(cond_operator, get_nu1());
//         preconditioner.compute(cond_operator);
//         update = preconditioner.solve(coef_seg);

//         /* Perform the update */
//         get_coefs().segment(D_phi_w_b, D_phi_w) =
//             get_coefs().segment(D_phi_w_b, D_phi_w) - step_size * update;

//         long int head_D_phi_w = D_phi_w_b;
//         long int tail_D_phi_w = get_coefs().size() - (D_phi_w_b + D_phi_w);
//         double shrink = (1 - step_size * get_nu1());
//         get_coefs().head(D_phi_w_b) =
//             get_coefs().head(D_phi_w_b) * shrink;
//         get_coefs().tail(tail_D_phi_w) =
//             get_coefs().tail(tail_D_phi_w) * shrink;
//     }

//     return *this;
// }

// #ifdef WRAP_PYTHON
//     auto py_fit_denseX_densey(const pybind11::array_t<double> & X,
//                               const pybind11::array_t<double> & y)
//         -> DivSOVK &;

//     auto py_fit_denseX_sparsey(const pybind11::array_t<double> & X,
//                                const Eigen::SparseMatrix<double> & y)
//         -> DivSOVK &;

//     auto py_fit_sparseX_densey(const Eigen::SparseMatrix<double> & X,
//                                const pybind11::array_t<double> & y)
//         -> DivSOVK &;

//     auto py_fit_sparseX_sparsey(const Eigen::SparseMatrix<double> & X,
//                                 const Eigen::SparseMatrix<double> & y)
//         -> DivSOVK &;

//     auto py_get_coefs() const
//         -> pybind11::array_t<double>;
// #endif

// };

#endif // OPT_HPP_INCLUDED