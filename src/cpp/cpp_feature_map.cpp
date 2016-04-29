#include <utility>
#include <iostream>
#include "feature_map.hpp"

using namespace Eigen;
using namespace std;
using namespace sitmo;

FeatureMap::FeatureMap(const feature_sampler_t & fs,
                       const feature_map_t & fm,
                       long int p, long int r) :
    _feature_sampler(fs),
    _feature_map(fm), _p(p), _r(r)
{

}

FeatureMap::FeatureMap(const FeatureMap & feature_map) :
    _feature_sampler(feature_map._feature_sampler),
    _feature_map(feature_map._feature_map),
    _p(feature_map._p), _r(feature_map._r)
{

}

static auto dec_gauss_rff(MatrixXd & B)
    -> function<void(const Ref<const MatrixXd>, // X
                     DecomposableLinOp<Eigen::MatrixXd,
                                       Eigen::MatrixXd> &, // phi_w
                     const Ref<const MatrixXd>, // W
                     Ref<MatrixXd>)> // Z
{
    return [B {std::move(B)}](
        const Ref<const MatrixXd> X,
        DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd> & phi_w,
        const Ref<const MatrixXd> W, Ref<MatrixXd> Z) mutable
    // Inputs: W     of size d x D
    //         X     of size d x n
    // Output: phi_w of size 2.r.D x p.n
    // Buffer: Z     of size D x n
    {
        long int n = X.cols();
        long int p = B.rows();
        // long int r = B.cols();
        long int D = W.cols();

        Z.noalias() = W.transpose() * X;
        phi_w.get_Lhs().block(0, 0, D, n).array() = Z.array().sin();
        phi_w.get_Lhs().block(D, 0, D, n).array() = Z.array().cos();
        phi_w.get_Lhs() /= sqrt(D);
        phi_w.set_Rhs(B);
    };
}

static inline auto thread_safe_normal_distribution(double mu, double sigma)
    -> function<double(prng_engine &)>
{
    return [mu, sigma](prng_engine & r_engine) {
        constexpr double epsilon = std::numeric_limits<double>::min();
        constexpr double two_pi = 2.0 * M_PI;

        double u1, u2;
        do {
           u1 = static_cast<double>(r_engine()) / r_engine.max();
           u2 = static_cast<double>(r_engine()) / r_engine.max();
        } while (u1 <= epsilon);
        double z0 = std::sqrt(-2. * std::log(u1)) * std::cos(two_pi * u2);
        return z0 * sigma + mu;
    };
}

auto DecomposableGaussian(const Ref<const MatrixXd> A, double sigma)
    -> FeatureMap
{
    JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU|Eigen::ComputeThinV);
    long int r = svd.rank();

    Eigen::MatrixXd B((svd.matrixU().leftCols(r) *
                       svd.singularValues().array().sqrt()
                          .matrix().head(r).asDiagonal()).transpose());

    return FeatureMap(
        thread_safe_normal_distribution(0, 1. / sigma),
        dec_gauss_rff(B), B.cols(), B.rows());
}