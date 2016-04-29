#include "opt.hpp"

using namespace Eigen;
using namespace std;
using namespace sitmo;

DSOVK::DSOVK(const loss_t & gloss,
             const feature_map_t & feature_map,
             const gamma_t & gamma, double nu, long int T,
             long int batch, long int block, long int T_cap, double cond) :
    _gloss(gloss),
    _phi(feature_map),
    _gamma(gamma),
    _nu(nu), _T(T), _T_cap(T_cap), _d(0), _p(0),
    _batch(batch), _block(block), _cond(cond),
    _coefs(2 * feature_map.r() * get_T_cap() * block)
{
    initParallel();
}

auto DSOVK::predict(const Ref<const MatrixXd> X,
                    Ref<MatrixXd> pred,
                    int nt, long int th)
    -> DSOVK &
{
    long int n = X.cols();
    long int batch = n > get_batch() ? get_batch() : n;
    long int block = get_block();
    long int n_batch = n > get_batch() ? n / get_batch(): 1;
    long int r = get_r();
    long int p =  get_p();
    pred.setZero();

    Eigen::Map<Eigen::VectorXd> pred_vview(pred.data(), pred.size());

#pragma omp parallel num_threads(nt) if(n * get_T() * r * block > th)
{
    prng_engine r_engine;

    DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>
        phi_w(get_feature_map().init(batch, block, get_p()));
    MatrixXd Z(block, batch);
    MatrixXd W(get_d(), block);

#pragma omp for
    for (long int i = 0; i < n_batch; ++i) {
        for (long int j = 0; j < get_T_cap(); ++j) {
            long int X_batch_b = (i * batch) % n;
            long int X_batch_e = min(X_batch_b + batch, n);
            long int c_batch = X_batch_e - X_batch_b;

            // get block random feature operator
            feature_map(X.middleCols(X_batch_b, c_batch), phi_w,
                        W, Z.leftCols(c_batch), r_engine, j);
            pred_vview.segment(X_batch_b * p, c_batch * p).noalias() +=
                phi_w.leftCols(c_batch).transpose() *
                get_coefs().segment(2 * r * j * block, 2 * r * block);
        }
    }
}
    return *this;
}

auto DSOVK::predict(const Ref<const MatrixXd> X,
                    int nt, long int th)
    -> MatrixXd
{
    MatrixXd pred(_p, X.cols());
    predict(X, pred, nt, th);
    return pred;
}