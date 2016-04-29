#ifndef LOSS_HPP_INCLUDED
#define LOSS_HPP_INCLUDED

#include <functional>

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#include "Eigen/Dense"

class Loss
{
public:

    using loss_t = std::function<void(
        const Eigen::Ref<const Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd>,
        Eigen::Ref<Eigen::MatrixXd>)>;

private:

    loss_t _loss;

public:
    Loss(const loss_t & loss);

    Loss(const Loss & loss);

    inline auto operator () (const Eigen::Ref<const Eigen::MatrixXd> pred,
                             const Eigen::Ref<const Eigen::MatrixXd> target,
                             Eigen::Ref<Eigen::MatrixXd> residuals)
        -> void
    {
        _loss(pred, target, residuals);
    }
};

auto RidgeLoss(void)
    -> Loss;

auto HingeLoss(double margin)
    -> Loss;

auto SoftMaxLoss(void)
    -> Loss;

#endif // LOSS_HPP_INCLUDED