#ifndef FEATURE_MAP_HPP_INCLUDED
#define FEATURE_MAP_HPP_INCLUDED

#include <functional>
#include <random>

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#include "Eigen/Dense"

#include "linear_operator.hpp"
#include "prng_engine.hpp"

class FeatureMap {
public:
    using feature_sampler_t = std::function<double(sitmo::prng_engine &)>;
    using feature_map_t = std::function<void(
        const Eigen::Ref<const Eigen::MatrixXd>, // X
        DecomposableLinOp<Eigen::MatrixXd,
                          Eigen::MatrixXd> &, // phi_w
        const Eigen::Ref<const Eigen::MatrixXd>, // W
        Eigen::Ref<Eigen::MatrixXd>)>; // Z

private:

    feature_sampler_t _feature_sampler;
    feature_map_t     _feature_map;
    long int          _r;
    long int          _p;

public:
    FeatureMap(const feature_sampler_t & fs,
               const feature_map_t & fm,
               long int p, long int r);

    FeatureMap(const FeatureMap & feature_map);

    inline auto r(void) const
        -> long int
    {
        return _r;
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
        _feature_map(X, phi_w, W, Z);
    }

};

auto DecomposableGaussian(const Eigen::Ref<const Eigen::MatrixXd> A,
                          double sigma)
    -> FeatureMap;

#endif // FEATURE_MAP_HPP_INCLUDED