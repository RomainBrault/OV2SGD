#ifndef LOSS_HPP_INCLUDED
#define LOSS_HPP_INCLUDED

#include <functional>

#ifdef RELEASE
#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#endif
#include "Eigen/Dense"
#include "Eigen/Sparse"

class Loss
{
public:

    using loss_dense_t = std::function<void(
        const Eigen::Ref<const Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd>,
        Eigen::Ref<Eigen::MatrixXd>)>;
    using loss_sparse_t = std::function<void(
        const Eigen::Ref<const Eigen::MatrixXd>,
        const Eigen::SparseMatrix<double>,
        Eigen::Ref<Eigen::MatrixXd>)>;

private:

    loss_dense_t _loss_dense;
    loss_sparse_t _loss_sparse;

public:
    Loss(const loss_dense_t & loss_dense, const loss_sparse_t & loss_sparse);

    Loss(const Loss & loss);

    inline auto operator () (const Eigen::Ref<const Eigen::MatrixXd> pred,
                             const Eigen::Ref<const Eigen::MatrixXd> target,
                             Eigen::Ref<Eigen::MatrixXd> residuals)
        -> void
    {
        _loss_dense(pred, target, residuals);
    }

    inline auto operator () (const Eigen::Ref<const Eigen::MatrixXd> pred,
                             const Eigen::SparseMatrix<double> & target,
                             Eigen::Ref<Eigen::MatrixXd> residuals)
        -> void
    {
        _loss_sparse(pred, target, residuals);
    }
};

auto SVRLoss(double eps)
    -> Loss;

auto MultitaskSVRLoss(double eps)
    -> Loss;

auto RidgeLoss(void)
    -> Loss;

auto MultitaskRidgeLoss(void)
    -> Loss;

auto HingeLoss(double margin)
    -> Loss;

auto SoftMaxLoss(void)
    -> Loss;

#endif // LOSS_HPP_INCLUDED