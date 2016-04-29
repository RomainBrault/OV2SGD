#include "loss.hpp"
#include <iostream>

using namespace Eigen;
using namespace std;

Loss::Loss(const loss_t & loss) :
    _loss(loss)
{

}

Loss::Loss(const Loss & loss) :
    _loss(loss._loss)
{

}

auto RidgeLoss(void)
    -> Loss
{
    return Loss([](const Ref<const MatrixXd> pred,
                   const Ref<const MatrixXd> target,
                   Ref<MatrixXd> residuals)
        -> void
    {
        residuals.noalias() = (pred - target) / target.cols();
    });
}

auto HingeLoss(double margin)
    -> Loss
{
    return Loss([margin](const Ref<const MatrixXd> pred,
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
    });
}

auto SoftMaxLoss(void)
    -> Loss
{
    return Loss([](const Ref<const MatrixXd> pred,
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
    });
}