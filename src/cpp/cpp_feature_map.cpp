#include <utility>
#include <iostream>
#include <cmath>
#include "feature_map.hpp"

using namespace Eigen;
using namespace std;
using namespace sitmo;

DivergenceFreeFeatureMap::DivergenceFreeFeatureMap(const feature_sampler_t & fs,
                       const feature_map_dense_t & fmd,
                       const feature_map_sparse_t & fms,
                       long int p) :
    _feature_sampler(fs),
    _feature_map_dense(fmd), _feature_map_sparse(fms),
    _p(p)
{

}

DivergenceFreeFeatureMap::DivergenceFreeFeatureMap(const DivergenceFreeFeatureMap & feature_map) :
    _feature_sampler(feature_map._feature_sampler),
    _feature_map_dense(feature_map._feature_map_dense),
    _feature_map_sparse(feature_map._feature_map_sparse),
    _p(feature_map._p)
{

}


DecomposableFeatureMap::DecomposableFeatureMap(const feature_sampler_t & fs,
                       const feature_map_dense_t & fmd,
                       const feature_map_sparse_t & fms,
                       long int p, long int r) :
    _feature_sampler(fs),
    _feature_map_dense(fmd), _feature_map_sparse(fms),
    _p(p), _r(r)
{

}

DecomposableFeatureMap::DecomposableFeatureMap(const DecomposableFeatureMap & feature_map) :
    _feature_sampler(feature_map._feature_sampler),
    _feature_map_dense(feature_map._feature_map_dense),
    _feature_map_sparse(feature_map._feature_map_sparse),
    _p(feature_map._p), _r(feature_map._r)
{

}

TransformableFeatureMap::TransformableFeatureMap(const feature_sampler_t & fs,
                       const feature_map_dense_t & fmd,
                       const feature_map_sparse_t & fms) :
    _feature_sampler(fs),
    _feature_map_dense(fmd), _feature_map_sparse(fms)
{

}

TransformableFeatureMap::TransformableFeatureMap(const TransformableFeatureMap & feature_map) :
    _feature_sampler(feature_map._feature_sampler),
    _feature_map_dense(feature_map._feature_map_dense),
    _feature_map_sparse(feature_map._feature_map_sparse)
{

}

static auto trans_dense_additive_rff(void)
    -> function<void(const Ref<const MatrixXd>, // X
                           Ref<      MatrixXd>, // phi_w
                     const Ref<const MatrixXd>, // W
                     Ref<MatrixXd>)> // Z
{
    return [](
        const Ref<const MatrixXd> X,
              Ref<      MatrixXd> phi_w,
        const Ref<const MatrixXd> W, Ref<MatrixXd> Z) mutable
    // Inputs: W     of size 1 x D
    //         X     of size d x n
    // Output: phi_w of size 2.r.D x p.n
    // Buffer: Z     of size D x n
    // assume d == p
    {
        long int n = X.cols();
        long int p = X.rows();
        // long int r = B.cols();
        long int D = W.cols();

        for (long int i = 0; i < p; ++i) {
            Z.noalias() = W.transpose() * X.block(i, 0, 1, n);
            for (long int j = 0; j < n; ++j) {
                for (long int k = 0; k < D; ++k) {
                    phi_w(    k, j * p + i) = sin(Z(k, j));
                    phi_w(D + k, j * p + i) = cos(Z(k, j));
                }
            }
        }
        phi_w /= sqrt(D);
    };
}

static auto dec_dense_additive_rff(const MatrixXd & B)
    -> function<void(const Ref<const MatrixXd>, // X
                     DecomposableLinOp<MatrixXd, MatrixXd> &, // phi_w
                     const Ref<const MatrixXd>, // W
                     Ref<MatrixXd>)> // Z
{
    return [B](
        const Ref<const MatrixXd> X,
        DecomposableLinOp<MatrixXd, MatrixXd> & phi_w,
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

static auto div_dense_additive_rff(void)
    -> function<void(const Ref<const MatrixXd>, // X
                     Ref<MatrixXd>, // phi_w
                     const Ref<const MatrixXd>, // W
                     Ref<MatrixXd>)> // Z
{
    return [](
        const Ref<const MatrixXd> X,
              Ref<      MatrixXd> phi_w,
        const Ref<const MatrixXd> W, Ref<MatrixXd> Z) mutable
    // Inputs: W     of size d x D
    //         X     of size d x n
    // Output: phi_w of size 2.(p - 1).D x p.n
    // Buffer: Z     of size D x n
    // assume d == p
    {
        long int n = X.cols();
        long int p = X.rows();
        // long int r = B.cols();
        long int D = W.cols();

        JacobiSVD<MatrixXd> svd(p, p, ComputeThinU|ComputeThinV);
        Z.noalias() = W.transpose() * X;
        for (long int k = 0; k < D; ++k) {
            svd.compute(W.col(k).squaredNorm() * Eigen::MatrixXd::Identity(p, p)
                        - W.col(k) * W.col(k).transpose());
            for (long int j = 0; j < n; ++j) {
                for (long int i = 0; i < p; ++i) {
                    for (long int l = 0; l < p - 1; ++l) {
                        phi_w((p - 1) * (    k) + l, j * p + i) =
                            sqrt(svd.singularValues()(l)) * svd.matrixV()(l, i)
                            * sin(Z(k, j));
                        phi_w((p - 1) * (D + k) + l, j * p + i) =
                            sqrt(svd.singularValues()(l)) * svd.matrixV()(l, i)
                            * cos(Z(k, j));
                    }
                }
            }
        }
        phi_w /= sqrt(D);
    };
}

static auto dec_dense_multiplicative_rff(const MatrixXd & B, double skewness)
    -> function<void(const Ref<const MatrixXd>, // X
                     DecomposableLinOp<MatrixXd, MatrixXd> &, // phi_w
                     const Ref<const MatrixXd>, // W
                     Ref<MatrixXd>)> // Z
{
    return [B, skewness](
        const Ref<const MatrixXd> X,
        DecomposableLinOp<MatrixXd, MatrixXd> & phi_w,
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

        Z.noalias() = W.transpose() * (X.array() + skewness).log().matrix();
        phi_w.get_Lhs().block(0, 0, D, n).array() = Z.array().sin();
        phi_w.get_Lhs().block(D, 0, D, n).array() = Z.array().cos();
        phi_w.get_Lhs() /= sqrt(D);
        phi_w.set_Rhs(B);
    };
}

static auto trans_sparse_additive_rff(void)
    -> function<void(const SparseMatrix<double> &, // X
                           Ref<      MatrixXd>,    // phi_w
                     const Ref<const MatrixXd>,    // W
                           Ref<      MatrixXd>)>   // Z
{
    return [](
        const SparseMatrix<double> & X,
              Ref<      MatrixXd> phi_w,
        const Ref<const MatrixXd> W,
              Ref<      MatrixXd> Z) mutable
    // Inputs: W     of size 1 x D
    //         X     of size d x n
    // Output: phi_w of size 2.r.D x p.n
    // Buffer: Z     of size D x n
    // assume d == p
    {
        long int n = X.cols();
        long int p = X.rows();
        // long int r = B.cols();
        long int D = W.cols();

        for (long int i = 0; i < p; ++i) {
            Z.noalias() = W.transpose() * X.block(i, 0, 1, n);
            for (long int j = 0; j < n; ++j) {
                for (long int k = 0; k < D; ++k) {
                    phi_w(    k, j * p + i) = sin(Z(k, j));
                    phi_w(D + k, j * p + i) = cos(Z(k, j));
                }
            }
        }
        phi_w /= sqrt(D);
    };
}

static auto dec_sparse_additive_rff(const MatrixXd & B)
    -> function<void(const SparseMatrix<double> &, // X
                     DecomposableLinOp<MatrixXd, MatrixXd> &, // phi_w
                     const Ref<const MatrixXd>, // W
                     Ref<MatrixXd>)> // Z
{
    return [B](
        const SparseMatrix<double> & X,
        DecomposableLinOp<MatrixXd, MatrixXd> & phi_w,
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

static auto div_sparse_additive_rff(void)
    -> function<void(const SparseMatrix<double> &, // X
                           Ref<      MatrixXd>, // phi_w
                     const Ref<const MatrixXd>, // W
                     Ref<MatrixXd>)> // Z
{
    return [](
        const SparseMatrix<double> X,
              Ref<      MatrixXd> phi_w,
        const Ref<const MatrixXd> W, Ref<MatrixXd> Z) mutable
    // Inputs: W     of size p x D
    //         X     of size d x n
    // Output: phi_w of size 2.(p - 1).D x p.n
    // Buffer: Z     of size D x n
    // assume d == p
    {
        long int n = X.cols();
        long int p = X.rows();
        // long int r = B.cols();
        long int D = W.cols();

        JacobiSVD<MatrixXd> svd(p, p, ComputeThinU|ComputeThinV);
        Z.noalias() = W.transpose() * X;
        for (long int k = 0; k < D; ++k) {
            svd.compute(W.col(k).squaredNorm() * Eigen::MatrixXd::Identity(p, p)
                        - W.col(k) * W.col(k).transpose());
            for (long int j = 0; j < n; ++j) {
                for (long int i = 0; i < p; ++i) {
                    for (long int l = 0; l < p - 1; ++l) {
                        phi_w(    (p - 1) * k + l, j * p + i) =
                            svd.matrixU()(l, i) * sin(Z(k, j));
                        phi_w(D + (p - 1) * k + l, j * p + i) =
                            svd.matrixU()(l, i) * cos(Z(k, j));
                    }
                }
            }
        }
        phi_w /= sqrt(D);
    };
}

static auto dec_sparse_multiplicative_rff(const MatrixXd & B, double skewness)
    -> function<void(const SparseMatrix<double> &, // X
                     DecomposableLinOp<MatrixXd, MatrixXd> &, // phi_w
                     const Ref<const MatrixXd>, // W
                     Ref<MatrixXd>)> // Z
{
    return [B, skewness](
        const SparseMatrix<double> & X,
        DecomposableLinOp<MatrixXd, MatrixXd> & phi_w,
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

        Z.noalias() = W.transpose() *
                      (X.toDense().array() + skewness).log().matrix();
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

static inline auto thread_safe_sech_distribution(void)
    -> function<double(prng_engine &)>
{
    return [](prng_engine & r_engine) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        return std::log(std::tan(M_PI / 2. * uniform(r_engine))) / M_PI;
    };
}

auto DivergenceFreeGaussian(double sigma, long int p)
    -> DivergenceFreeFeatureMap
{
    return DivergenceFreeFeatureMap(
        thread_safe_normal_distribution(0, 1. / sigma),
        div_dense_additive_rff(), div_sparse_additive_rff(), p);
}

auto TransformableGaussian(double sigma)
    -> TransformableFeatureMap
{
    return TransformableFeatureMap(
        thread_safe_normal_distribution(0, 1. / sigma),
        trans_dense_additive_rff(), trans_sparse_additive_rff());
}

auto DecomposableGaussian(const Ref<const MatrixXd> A, double sigma)
    -> DecomposableFeatureMap
{
    JacobiSVD<MatrixXd> svd(A, ComputeThinU|ComputeThinV);
    long int r = svd.rank();

    MatrixXd B((svd.matrixU().leftCols(r) *
                       svd.singularValues().array().sqrt()
                          .matrix().head(r).asDiagonal()).transpose());

    return DecomposableFeatureMap(
        thread_safe_normal_distribution(0, 1. / sigma),
        dec_dense_additive_rff(B), dec_sparse_additive_rff(B),
        B.cols(), B.rows());
}

auto DecomposableGaussianB(const Ref<const MatrixXd> B, double sigma)
    -> DecomposableFeatureMap
{
    return DecomposableFeatureMap(
        thread_safe_normal_distribution(0, 1. / sigma),
        dec_dense_additive_rff(B), dec_sparse_additive_rff(B),
        B.cols(), B.rows());
}

auto DecomposableSkewedChi2(const Ref<const MatrixXd> A, double skewness)
    -> DecomposableFeatureMap
{
    JacobiSVD<MatrixXd> svd(A, ComputeThinU|ComputeThinV);
    long int r = svd.rank();

    MatrixXd B((svd.matrixU().leftCols(r) *
                       svd.singularValues().array().sqrt()
                          .matrix().head(r).asDiagonal()).transpose());

    return DecomposableFeatureMap(
        thread_safe_sech_distribution(),
        dec_dense_multiplicative_rff(B, skewness),
        dec_sparse_multiplicative_rff(B, skewness),
        B.cols(), B.rows());
}

auto DecomposableSkewedChi2B(const Ref<const MatrixXd> B, double skewness)
    -> DecomposableFeatureMap
{
    return DecomposableFeatureMap(
        thread_safe_sech_distribution(),
        dec_dense_multiplicative_rff(B, skewness),
        dec_sparse_multiplicative_rff(B, skewness),
        B.cols(), B.rows());
}