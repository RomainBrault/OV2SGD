#include "loss.hpp"
#include <iostream>
#include <memory>

using namespace Eigen;
using namespace std;

Loss::Loss(const loss_dense_t & loss_dense,
           const loss_sparse_t & loss_sparse) :
    _loss_dense(loss_dense), _loss_sparse(loss_sparse)
{

}

Loss::Loss(const Loss & loss) :
    _loss_dense(loss._loss_dense), _loss_sparse(loss._loss_sparse)
{

}

auto SVRLoss(double eps)
    -> Loss
{
    return Loss(
        [eps](const Ref<const MatrixXd> pred,
              const Ref<const MatrixXd> target,
              Ref<MatrixXd> residuals)
            -> void
        {
            for (long int i = 0; i < pred.cols(); ++i) {
                for (long int j = 0; j < pred.rows(); ++j) {
                    double res = pred(j, i) - target(j, i);
                    double res_t = res > eps ? 1 : (res < -eps ? -1 : 0);
                    residuals(j, i) = res_t;
                }
            }
            residuals /= target.cols();
        },

        [eps](const Ref<const MatrixXd> pred,
              const SparseMatrix<double> & target,
              Ref<MatrixXd> residuals)
            -> void
        {
            for (long int i = 0; i < pred.cols(); ++i) {
                for (long int j = 0; j < pred.rows(); ++j) {
                    double res = pred(j, i) - target.coeff(j, i);
                    double res_t = res > eps ? 1 : (res < -eps ? -1 : 0);
                    residuals(j, i) = res_t;
                }
            }
            residuals /= target.cols();
        }
    );
}

auto MultitaskSVRLoss(double eps)
    -> Loss
{
    return Loss(
        [eps](const Ref<const MatrixXd> pred,
              const Ref<const MatrixXd> target,
              Ref<MatrixXd> residuals)
            -> void
        {
            residuals.setZero();
            for (long int i = 0; i < pred.cols(); ++i) {
                long int idx = static_cast<long int>(target(1, i));
                double res =  pred(idx, i) - target(0, i);
                double res_t = res > eps ? 1 : (res < -eps ? -1 : 0);
                residuals(idx, i) = res_t;
            }
            residuals /= target.cols();
        },

        [eps](const Ref<const MatrixXd> pred,
              const SparseMatrix<double> & target,
              Ref<MatrixXd> residuals)
            -> void
        {
            residuals.setZero();
            for (long int i = 0; i < pred.cols(); ++i) {
                long int idx = static_cast<long int>(target.coeff(1, i));
                double res = pred(idx, i) - target.coeff(0, i);
                double res_t = res > eps ? 1 : (res < -eps ? -1 : 0);
                residuals(idx, i) = res_t;
            }
            residuals /= target.cols();
        }
    );
}

auto RidgeLoss(void)
    -> Loss
{
    return Loss(
        [](const Ref<const MatrixXd> pred,
           const Ref<const MatrixXd> target,
           Ref<MatrixXd> residuals)
            -> void
        {
            residuals.noalias() = (pred - target) / target.cols();
        },

        [](const Ref<const MatrixXd> pred,
           const SparseMatrix<double> & target,
           Ref<MatrixXd> residuals)
            -> void
        {
            residuals = pred;
            residuals -= target;
            residuals /= target.cols();
        }
    );
}

auto MultitaskRidgeLoss(void)
    -> Loss
{
    return Loss(
        [](const Ref<const MatrixXd> pred,
           const Ref<const MatrixXd> target,
           Ref<MatrixXd> residuals)
            -> void
        {
            residuals.setZero();
            for (long int i = 0; i < pred.cols(); ++i) {
                long int idx = static_cast<long int>(target(1, i));
                residuals(idx, i) = pred(idx, i) - target(0, i);
            }
            residuals /= target.cols();
        },

        [](const Ref<const MatrixXd> pred,
           const SparseMatrix<double> & target,
           Ref<MatrixXd> residuals)
            -> void
        {
            residuals.setZero();
            for (long int i = 0; i < pred.cols(); ++i) {
                long int idx = static_cast<long int>(target.coeff(1, i));
                residuals(idx, i) = pred(idx, i) - target.coeff(0, i);
            }
            residuals /= target.cols();
        }
    );
}

auto HingeLoss(double margin)
    -> Loss
{
    return Loss(
        [margin](const Ref<const MatrixXd> pred,
                 const Ref<const MatrixXd> target,
                 Ref<MatrixXd> residuals)
        -> void
        {
            for (long int i = 0; i < pred.cols(); ++i) {
                long int p; target.col(i).maxCoeff(&p); // true label index
                double max_err = 0;
                for (long int j = 0; j < pred.rows(); ++j) {
                    double dist = pred(j, i) - pred(p, i);
                    double cost = 1 - (target(j, i) * target(p, i));
                    max_err = std::max(cost - dist, max_err);
                }
                residuals.col(i).setZero();
                if (max_err > 0) {
                    residuals(p, i) = -target(p, i);
                }
            }
            residuals /= target.cols();
        },
        [margin](const Ref<const MatrixXd> pred,
                 const SparseMatrix<double> & target,
                 Ref<MatrixXd> residuals)
        -> void
        {
            for (long int i = 0; i < pred.cols(); ++i) {
                using iter_t = typename SparseMatrix<double>::InnerIterator;
                iter_t it(target, i);
                long int p = it.index(); // first non zero index is good

                double max_err = 0;
                for (long int j = 0; j < pred.rows(); ++j) {
                    double dist = pred(j, i) - pred(p, i);
                    double cost = 1 - (target.coeff(j, i) *
                                       target.coeff(p, i));
                    max_err = std::max(cost - dist, max_err);
                }
                residuals.col(i).setZero();
                if (max_err > 0) {
                    residuals(p, i) = -target.coeff(p, i);
                }
            }
            residuals /= target.cols();
        }
    );
}

auto SoftMaxLoss(void)
    -> Loss
{
    return Loss(
        [](const Ref<const MatrixXd> pred,
           const Ref<const MatrixXd> target,
           Ref<MatrixXd> residuals)
        -> void
        {
            for (long int i = 0; i < pred.cols(); ++i) {
                residuals.col(i) =
                    (pred.col(i).array() - pred.col(i).maxCoeff()).exp();
                residuals.col(i) =
                    residuals.col(i).array() / residuals.col(i).sum();
            }
            residuals = (residuals - target) / target.cols();
        },
        [](const Ref<const MatrixXd> pred,
           const SparseMatrix<double> & target,
           Ref<MatrixXd> residuals)
        -> void
        {
            for (long int i = 0; i < pred.cols(); ++i) {
                residuals.col(i) =
                    (pred.col(i).array() - pred.col(i).maxCoeff()).exp();
                residuals.col(i) =
                    residuals.col(i).array() / residuals.col(i).sum();
            }
            residuals -= target;
            residuals /= target.cols();
        }
    );
}

// auto MultitaskNoveltyLoss(double rate)
//     -> Loss
// {
//     auto tau = std::make_shared<double>(0);
//     return Loss(
//         [tau, rate](const Ref<const MatrixXd> pred,
//            const Ref<const MatrixXd> target,
//            Ref<MatrixXd> residuals)
//         -> void
//         {
//             for (long int i = 0; i < pred.cols(); ++i) {
//                 for (long int j = 0; j < pred.rows(); ++j) {
//                     if (pred(j, i) < tau) {

//                     }
//                 }
//             }
//             residuals /= target.cols();
//         },
//         [tau, rate](const Ref<const MatrixXd> pred,
//            const SparseMatrix<double> & target,
//            Ref<MatrixXd> residuals)
//         -> void
//         {

//         }
//     );
// }