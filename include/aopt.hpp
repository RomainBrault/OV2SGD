// #ifndef AOPT_HPP_INCLUDED
// #define AOPT_HPP_INCLUDED

// #include <iostream>
// #include <iomanip>
// #include <utility>
// #include <algorithm>
// #include <chrono>
// #include <limits>

// #ifdef RELEASE
// #define EIGEN_NO_AUTOMATIC_RESIZING
// #define EIGEN_NO_DEBUG
// #endif
// #include "Eigen/Dense"
// // #include "Eigen/IterativeLinearSolvers"
// #include "unsupported/Eigen/IterativeSolvers"

// #ifdef WRAP_PYTHON
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #endif // WRAP_PYTHON

// #include "loss.hpp"
// #include "feature_map.hpp"
// #include "learning_rate.hpp"
// #include "prng_engine.hpp"

// class ADSOVK {

// public:
//     using loss_t = Loss;

//     using feature_map_t = DecomposableFeatureMap;

//     using gamma_t = LearningRate::learning_rate_t;
//     using mu_t = LearningRate::learning_rate_t;

//     using feature_sampler_t =  DecomposableFeatureMap::feature_sampler_t;

// private:
//     loss_t        _gloss;
//     feature_map_t _phi;
//     mu_t          _mu;
//     gamma_t       _gamma;

//     double  _nu;

//     long int _T;
//     long int _T_cap;
//     long int _d;
//     long int _p;
//     long int _batch;
//     long int _block;

//     Eigen::VectorXd _coefs;
//     Eigen::VectorXd _coefs_avg;

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
//         long int r = get_r();
//         long int D = W.cols();

//         Eigen::Map<Eigen::VectorXd> pred_vview(pred.data(), pred.size());

//         for (long int j = 0; j < t; ++j) {
//             feature_map(X, phi_w, W, Z.leftCols(n), r_engine, j);
//             pred.noalias() = pred +
//                 phi_w.leftCols(n).transpose() *
//                 get_coefs().segment(2 * r * j * _block, 2 * r * _block);
//         }
//         return pred;
//     }

// public:
//     ADSOVK(const loss_t & gloss,
//            const feature_map_t & fm,
//            const gamma_t & gamma,
//            const mu_t & mu,
//            double nu, long int T,
//            long int p,
//            long int batch = 100, long int block = 100, long int T_cap = -1);

// private:

//     inline auto set_batch(long int val)
//         -> ADSOVK &
//     {
//         _batch = val;
//         return *this;
//     }

//     inline auto set_d(long int val)
//         -> ADSOVK &
//     {
//         _d = val;
//         return *this;
//     }

//     inline auto set_p(long int val)
//         -> ADSOVK &
//     {
//         _p = val;
//         return *this;
//     }

//     inline auto get_coefs(void)
//         -> Eigen::VectorXd &
//     {
//         return _coefs;
//     }

//     inline auto get_coefs_avg(void)
//         -> Eigen::VectorXd &
//     {
//         return _coefs_avg;
//     }

// public:

//     inline auto get_d(void) const
//         -> long int
//     {
//         return _d;
//     }

//     inline auto get_r(void) const
//         -> long int
//     {
//         return get_feature_map().r();
//     }

//     inline auto get_p(void) const
//         -> long int
//     {
//         return _p;
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

//     inline auto get_nu(void) const
//         -> double
//     {
//         return _nu;
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
//         -> const DecomposableFeatureMap &
//     {
//         return _phi;
//     }

//     inline auto get_coefs(void) const
//         -> const Eigen::VectorXd &
//     {
//         return _coefs;
//     }

//     inline auto get_coefs_avg(void) const
//         -> const Eigen::VectorXd &
//     {
//         return _coefs_avg;
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

//     inline auto averaging_rate(long int t)
//         -> double
//     {
//         return _mu(t);
//     }

//     template <typename M1, typename M2>
//     auto predict(const M1 & X, M2 & pred, int nt, long int th)
//         -> ADSOVK &
//     {
//         Eigen::initParallel();
//         Eigen::setNbThreads(1);

//         long int n = X.cols();
//         long int batch = n > get_batch() ? get_batch() : n;
//         long int block = get_block();
//         long int n_batch = n > get_batch() ? n / get_batch(): 1;
//         long int r = get_r();
//         long int p =  get_p();
//         pred.setZero();

//     #pragma omp parallel num_threads(nt) if(n * get_T() * r * block > th)
//     {
//         sitmo::prng_engine r_engine;

//         DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>
//             phi_w(get_feature_map().init(batch, block, get_p()));
//         Eigen::MatrixXd Z(block, batch);
//         Eigen::MatrixXd W(get_d(), block);

//     #pragma omp for
//         for (long int i = 0; i < n_batch + 1; ++i) {
//             for (long int j = 0; j < get_T_cap(); ++j) {
//                 long int X_batch_b = (i * batch) % n;
//                 long int X_batch_e = std::min(X_batch_b + batch, n);
//                 long int c_batch = X_batch_e - X_batch_b;
//                 // get block random feature operator
//                 feature_map(X.middleCols(X_batch_b, c_batch), phi_w,
//                             W, Z.leftCols(c_batch), r_engine, j);
//                 pred.middleCols(X_batch_b, c_batch).noalias() =
//                     pred.middleCols(X_batch_b, c_batch) +
//                     phi_w.leftCols(c_batch).transpose() *
//                     get_coefs_avg().segment(2 * r * j * block, 2 * r * block);
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

// template <typename M1, typename M2>
// auto fit(const M1 & X, const M2 & y)
//     -> ADSOVK &
// {
//     set_d(X.rows());
//     set_batch(std::min<typename M1::Index>(get_batch(), X.cols()));

//     long int r = get_r();
//     long int d = get_d();
//     long int p = get_p();
//     long int D_w = get_block();
//     long int D_phi_w = 2 * r * D_w;
//     long int m_batch = get_batch();
//     long int T = get_T();

//     get_coefs().setZero();
//     get_coefs_avg().setZero();

//     sitmo::prng_engine r_engine;

//     Eigen::MatrixXd W(d, D_w);
//     Eigen::MatrixXd pred(p, m_batch);
//     Eigen::MatrixXd residues(p, m_batch);
//     Eigen::VectorXd update(D_phi_w);
//     Eigen::MatrixXd Z(D_w, m_batch);

//     DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>
//         phi_w(get_feature_map().init(get_batch(), get_block(), get_p()));

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
//         double avg_rate = averaging_rate(t);
//         update.noalias() = - step_size * (
//             phi_w.leftCols(c_batch) * grad_vview) +
//             get_nu() * get_coefs().segment(D_phi_w_b, D_phi_w);

//         /* Perform the update */
//         get_coefs().segment(D_phi_w_b, D_phi_w) =
//             get_coefs().segment(D_phi_w_b, D_phi_w) + update;

//         long int head_D_phi_w = D_phi_w_b;
//         long int tail_D_phi_w = get_coefs().size() - (D_phi_w_b + D_phi_w);
//         double shrink = (1 - step_size * get_nu());
//         get_coefs().head(D_phi_w_b) =
//             get_coefs().head(D_phi_w_b) * shrink;
//         get_coefs().tail(tail_D_phi_w) =
//             get_coefs().tail(tail_D_phi_w) * shrink;

//         /* Maintain averaging */
//         get_coefs_avg() =
//             get_coefs_avg() + avg_rate * (get_coefs() - get_coefs_avg());
//     }

//     return *this;
// }

// #ifdef WRAP_PYTHON
//     auto py_fit_denseX_densey(const pybind11::array_t<double> & X,
//                               const pybind11::array_t<double> & y)
//         -> ADSOVK &;

//     auto py_fit_denseX_sparsey(const pybind11::array_t<double> & X,
//                                const Eigen::SparseMatrix<double> & y)
//         -> ADSOVK &;

//     auto py_fit_sparseX_densey(const Eigen::SparseMatrix<double> & X,
//                                const pybind11::array_t<double> & y)
//         -> ADSOVK &;

//     auto py_fit_sparseX_sparsey(const Eigen::SparseMatrix<double> & X,
//                                 const Eigen::SparseMatrix<double> & y)
//         -> ADSOVK &;

// #endif

// };

// #endif // AOPT_HPP_INCLUDED