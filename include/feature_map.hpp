#ifndef FEATURE_MAP_HPP_INCLUDED
#define FEATURE_MAP_HPP_INCLUDED

#include <functional>
#include <random>

#ifdef NDEBUG
#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#endif
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "linear_operator.hpp"
#include "prng_engine.hpp"

class DivergenceFreeFeatureMap {
public:
    using feature_sampler_t = std::function<double(sitmo::prng_engine &)>;
    using feature_map_dense_t = std::function<
        void(const Eigen::Ref<const Eigen::MatrixXd>,               // X
                   Eigen::Ref<      Eigen::MatrixXd>,               // phi_w
             const Eigen::Ref<const Eigen::MatrixXd>,               // W
                   Eigen::Ref<      Eigen::MatrixXd>)>;             // Z
    using feature_map_sparse_t = std::function<
        void(const Eigen::SparseMatrix<double> &,                   // X
                   Eigen::Ref<      Eigen::MatrixXd>,               // phi_w
             const Eigen::Ref<const Eigen::MatrixXd>,               // W
                   Eigen::Ref<      Eigen::MatrixXd>)>;             // Z

private:

    feature_sampler_t    _feature_sampler;
    feature_map_dense_t  _feature_map_dense;
    feature_map_sparse_t _feature_map_sparse;
    long int             _p;

public:
    DivergenceFreeFeatureMap(const feature_sampler_t & fs,
                             const feature_map_dense_t & fmd,
                             const feature_map_sparse_t & fms,
                             long int p);

    DivergenceFreeFeatureMap(const DivergenceFreeFeatureMap & feature_map);

    inline auto r(void) const
        -> long int
    {
        return _p - 1;
    }

    inline auto p(void) const
        -> long int
    {
        return _p;
    }

    inline auto sampler(sitmo::prng_engine & r_engine)
        -> double
    {
        return _feature_sampler(r_engine);
    }

    inline auto init(long int batch, long int block, long int p) const
        -> Eigen::MatrixXd
    {
        return Eigen::MatrixXd(2 * r() * block, batch * p);
    }

    inline auto operator ()(const Eigen::Ref<const Eigen::MatrixXd> X,
                                  Eigen::Ref<      Eigen::MatrixXd> phi_w,
                                  Eigen::Ref<      Eigen::MatrixXd> W,
                                  Eigen::Ref<      Eigen::MatrixXd> Z,
                            sitmo::prng_engine & r_engine)
        -> void
    {
        long int D = W.cols();
        long int d = W.rows();
        for (long int j = 0; j < D; ++j) {
            for (long int i = 0; i < d; ++i) {
                W(i, j) = sampler(r_engine);
            }
        }
    }

    inline auto operator ()(const Eigen::SparseMatrix<double> & X,
                                  Eigen::Ref<Eigen::MatrixXd> phi_w,
                                  Eigen::Ref<Eigen::MatrixXd> W,
                                  Eigen::Ref<Eigen::MatrixXd> Z,
                            sitmo::prng_engine & r_engine)
        -> void
    {
        long int D = W.cols();
        long int d = W.rows();
        for (long int j = 0; j < D; ++j) {
            for (long int i = 0; i < d; ++i) {
                W(i, j) = sampler(r_engine);
            }
        }
        _feature_map_sparse(X, phi_w, W, Z);
    }

};

class TransformableFeatureMap {
public:
    using feature_sampler_t = std::function<double(sitmo::prng_engine &)>;
    using feature_map_dense_t = std::function<
        void(const Eigen::Ref<const Eigen::MatrixXd>, // X
                   Eigen::Ref<      Eigen::MatrixXd>, // phi_w
             const Eigen::Ref<const Eigen::MatrixXd>, // W
                   Eigen::Ref<      Eigen::MatrixXd>)>; // Z
    using feature_map_sparse_t = std::function<
        void(const Eigen::SparseMatrix<double> &, // X
                   Eigen::Ref<      Eigen::MatrixXd>, // phi_w
             const Eigen::Ref<const Eigen::MatrixXd>, // W
                   Eigen::Ref<      Eigen::MatrixXd>)>; // Z

private:

    feature_sampler_t    _feature_sampler;
    feature_map_dense_t  _feature_map_dense;
    feature_map_sparse_t _feature_map_sparse;

public:
    TransformableFeatureMap(const feature_sampler_t & fs,
                            const feature_map_dense_t & fmd,
                            const feature_map_sparse_t & fms);

    TransformableFeatureMap(const TransformableFeatureMap & feature_map);

    inline auto sampler(sitmo::prng_engine & r_engine)
        -> double
    {
        return _feature_sampler(r_engine);
    }

    inline auto init(long int batch, long int block, long int p) const
        -> Eigen::MatrixXd
    {
        return Eigen::MatrixXd(2 * block, batch * p);
    }

    inline auto operator ()(const Eigen::Ref<const Eigen::MatrixXd> X,
                            Eigen::Ref<Eigen::MatrixXd> phi_w,
                            Eigen::Ref<Eigen::MatrixXd> W,
                            Eigen::Ref<Eigen::MatrixXd> Z,
                            sitmo::prng_engine & r_engine)
        -> void
    {
        long int D = W.cols();
        for (long int j = 0; j < D; ++j) {
            W(0, j) = sampler(r_engine);
        }
        _feature_map_dense(X, phi_w, W, Z);
    }

    inline auto operator ()(const Eigen::SparseMatrix<double> & X,
                            Eigen::Ref<Eigen::MatrixXd> phi_w,
                            Eigen::Ref<Eigen::MatrixXd> W,
                            Eigen::Ref<Eigen::MatrixXd> Z,
                            sitmo::prng_engine & r_engine)
        -> void
    {
        long int D = W.cols();
        for (long int j = 0; j < D; ++j) {
            W(0, j) = sampler(r_engine);
        }
        _feature_map_sparse(X, phi_w, W, Z);
    }

};

class DecomposableFeatureMap {
public:
    using feature_sampler_t = std::function<double(sitmo::prng_engine &)>;
    using feature_map_dense_t = std::function<
        void(const Eigen::Ref<const Eigen::MatrixXd>,               // X
             DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd> &, // phi_w
             const Eigen::Ref<const Eigen::MatrixXd>,               // W
                   Eigen::Ref<      Eigen::MatrixXd>)>;             // Z
    using feature_map_sparse_t = std::function<
        void(const Eigen::SparseMatrix<double> &,                   // X
             DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd> &, // phi_w
             const Eigen::Ref<const Eigen::MatrixXd>,               // W
                   Eigen::Ref<      Eigen::MatrixXd>)>;             // Z

private:

    feature_sampler_t    _feature_sampler;
    feature_map_dense_t  _feature_map_dense;
    feature_map_sparse_t _feature_map_sparse;
    long int             _r;
    long int             _p;

public:
    DecomposableFeatureMap(const feature_sampler_t & fs,
                           const feature_map_dense_t & fmd,
                           const feature_map_sparse_t & fms,
                           long int p, long int r);

    DecomposableFeatureMap(const DecomposableFeatureMap & feature_map);

    inline auto r(void) const
        -> long int
    {
        return _r;
    }

    inline auto p(void) const
        -> long int
    {
        return _p;
    }

    inline auto sampler(sitmo::prng_engine & r_engine)
        -> double
    {
        return _feature_sampler(r_engine);
    }

    inline auto init(long int batch, long int block, long int p) const
        -> DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>
    {
        return DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>(2 * block,
                                                                   batch,
                                                                   r(),
                                                                   p);
    }

    inline auto operator ()(const Eigen::Ref<const Eigen::MatrixXd> X,
                            DecomposableLinOp<Eigen::MatrixXd,
                                              Eigen::MatrixXd> & phi_w,
                            Eigen::Ref<Eigen::MatrixXd> W,
                            Eigen::Ref<Eigen::MatrixXd> Z,
                            sitmo::prng_engine & r_engine)
        -> void
    {
        long int D = W.cols();
        long int d = W.rows();
        for (long int j = 0; j < D; ++j) {
            for (long int i = 0; i < d; ++i) {
                W(i, j) = sampler(r_engine);
            }
        }
        _feature_map_dense(X, phi_w, W, Z);
    }

    inline auto operator ()(const Eigen::SparseMatrix<double> & X,
                            DecomposableLinOp<Eigen::MatrixXd,
                                              Eigen::MatrixXd> & phi_w,
                            Eigen::Ref<Eigen::MatrixXd> W,
                            Eigen::Ref<Eigen::MatrixXd> Z,
                            sitmo::prng_engine & r_engine)
        -> void
    {
        long int D = W.cols();
        long int d = W.rows();
        for (long int j = 0; j < D; ++j) {
            for (long int i = 0; i < d; ++i) {
                W(i, j) = sampler(r_engine);
            }
        }
        _feature_map_sparse(X, phi_w, W, Z);
    }

};

auto DivergenceFreeGaussian(double sigma, long int p)
    -> DivergenceFreeFeatureMap;

auto TransformableGaussian(double sigma)
    -> TransformableFeatureMap;

auto DecomposableGaussian(const Eigen::Ref<const Eigen::MatrixXd> A,
                          double sigma)
    -> DecomposableFeatureMap;

auto DecomposableGaussianB(const Eigen::Ref<const Eigen::MatrixXd> B,
                           double sigma)
    -> DecomposableFeatureMap;

auto DecomposableSkewedChi2(const Eigen::Ref<const Eigen::MatrixXd> A,
                            double skewness)
    -> DecomposableFeatureMap;

auto DecomposableSkewedChi2B(const Eigen::Ref<const Eigen::MatrixXd> B,
                             double skewness)
    -> DecomposableFeatureMap;

#endif // FEATURE_MAP_HPP_INCLUDED