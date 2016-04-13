#ifndef OPT_HPP_INCLUDED
#define OPT_HPP_INCLUDED

#if WRAP_PYTHON
#include <Python.h>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#endif

#include <iostream>
#include <random>
#include <functional>
#include <type_traits>
#include <tuple>
#include <Eigen/Dense>

template <typename T>
auto inline sqr(T x)
    -> T
{
    return x * x;
}

auto DecomposableGaussian(const Eigen::Ref<const Eigen::MatrixXd> A,
                          double sigma, long int seed = 0)
    -> std::tuple<std::function<double()>,
                  std::function<void(const Eigen::Ref<const Eigen::MatrixXd>,
                                     const Eigen::Ref<const Eigen::MatrixXd>,
                                     Eigen::Ref<Eigen::MatrixXd>,
                                     Eigen::Ref<Eigen::MatrixXd>)>,
                  long int>;

auto inline ridge_loss(const Eigen::Ref<const Eigen::MatrixXd> pred,
                       const Eigen::Ref<const Eigen::MatrixXd> target,
                       Eigen::Ref<Eigen::MatrixXd> residuals)
    -> void
{
    residuals.noalias() = pred - target;
}

class DSOVK {

public:
    using loss_type = std::function<void(
        const Eigen::Ref<const Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd>,
        Eigen::Ref<Eigen::MatrixXd>)>;

    using phi_type = std::function<void(
        const Eigen::Ref<const Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd>,
        Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>)>;

    using gamma_type = std::function<double(int)>;

    using feature_sampler_type = std::function<double()>;

    using feature_map = std::tuple<feature_sampler_type, phi_type, long int>;

private:
    loss_type            _gloss;
    phi_type             _phi;
    feature_sampler_type _feature_sampler;

    gamma_type _gamma;
    double     _nu;

    long int _T;
    long int _t;
    long int _r;
    long int _d;
    long int _p;
    long int _batch;
    long int _block;

    double _cond;

    Eigen::VectorXd _coefs;
    Eigen::MatrixXd _W;

private:
    static inline auto _condition(Eigen::Ref<Eigen::MatrixXd> mat, double lbda)
        -> void;

    auto _predict(const Eigen::Ref<const Eigen::MatrixXd> X,
                  Eigen::Ref<Eigen::MatrixXd> phi_w,
                  Eigen::Ref<Eigen::MatrixXd> pred,
                  Eigen::Ref<Eigen::MatrixXd> Z)
        -> Eigen::MatrixXd;
    // Inputs: X     of size d x n
    //         phi_w of size p.n x 2.r.D
    // Output: pred  of size p x n initialized to 0

public:
    DSOVK(const loss_type & gloss,
          const feature_map & fm,
          const gamma_type & gamma, double nu, long int T,
          long int batch = 10, long int block = 10, double cond = 0);
public:
    auto predict(const Eigen::Ref<const Eigen::MatrixXd> X,
                 Eigen::Ref<Eigen::MatrixXd> pred,
                 int nt = 8, long int th = 10000)
        -> Eigen::MatrixXd;

    auto predict(const Eigen::Ref<const Eigen::MatrixXd> X)
        -> Eigen::MatrixXd;

    auto fit(const Eigen::Ref<const Eigen::MatrixXd> X,
             const Eigen::Ref<const Eigen::MatrixXd> y)
        -> DSOVK &;
        // Precondition: X are shuffled

};

#endif // OPT_HPP_INCLUDED