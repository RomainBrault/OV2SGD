// #define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_NO_MALLOC

#include <iostream>
// #include <cmath>
// #include <type_traits>
// #include <functional>
// #include <algorithm>
// #include "opt.hpp"
#include "linear_operator.hpp"

// using namespace Eigen;

    // std::function<Eigen::MatrixXd(auto)> F(int count)
    // {
    //     return [](auto m) { return Eigen::MatrixXd::Random(m, m); };
    // }

auto main(int argc, char* argv[]) -> int
{
    // Eigen::MatrixXd A = Eigen::MatrixXd::Random(1000, 10000);
    // Eigen::MatrixXd B = Eigen::MatrixXd::Random(500, 500);
    // Eigen::MatrixXd X = Eigen::MatrixXd::Random(500000, 1);
    // // std::cout << X << std::endl;

    // auto C = DecomposableLinOp<Eigen::MatrixXd, Eigen::MatrixXd>(std::move(A), std::move(B));

    // Eigen::MatrixXd Y(500000, 1);
    // Y.noalias() = C.leftCols(1000).transpose() * X;

    srand(0);
    Eigen::VectorXd A = Eigen::VectorXd::Random(10);
    Eigen::Map<Eigen::MatrixXd> M(A.data(), 5, 2);
    std::cout << M << std::endl;

    std::cout << std::endl;
    srand(0);
    Eigen::Map<Eigen::MatrixXd, 0,
               Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        M2(A.data(), 5, 2,
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, 5));
    std::cout << M2 << std::endl;

    // std::cout << X - Y << std::endl;
    // int n = 1024 * 10;
    // int d = 2;
    // int p = 1;
    // int r = 1;

    // Eigen::MatrixXd X = 5 * Eigen::MatrixXd::Random(d, n);
    // Eigen::MatrixXd y = (0.5 * M_PI * X.colwise().norm()).array().cos() *
    //                     (-0.1 * M_PI * X.colwise().norm()).array().exp();

    // // for (auto i = 0; i < X.cols(); ++i) {
    // //     X.col(i).array() /= X.col(i).squaredNorm();
    // // }
    // // X.colwise().normalize();


    // // Eigen::MatrixXd D(X.cols(), X.cols());
    // // for (auto i = 0; i < X.cols(); i++)
    // //     D.col(i) = (X.colwise() - X.col(i)).colwise().norm().transpose();

    // // std::nth_element(D.data(),
    // //                  D.data() + D.size()/2,
    // //                  D.data());
    // // double sigma = 0.1 * D.data()[D.size()/2];
    // // std::cout << sigma << std::endl;

    // int T = 1;
    // int batch = 1024;
    // int block = 1024;
    // double gamma0 = .5;
    // double lbda = 1e-6;
    // // std::mt19937 r_engine(0);
    // // auto feature_sampler = std::bind(std::normal_distribution<>(0, 100.), r_engine);
    // // auto feature_map = dec_rff(Eigen::MatrixXd::Identity(1, 1), batch, block);
    // auto feature_map = DecomposableGaussian(Eigen::MatrixXd::Identity(1, 1), 1.09354);
    // // auto gamma = [gamma0, lbda](int i) -> double {return gamma0 / (1. + gamma0 * lbda * i);};
    // auto gamma = [gamma0, lbda](int i) -> double {return .1 / (1 + 1e-3 * i);};

    // DSOVK estimator(RidgeLoss(), feature_map, gamma, lbda, T, batch, block);


    // estimator.fit(X, y);

    // std::cout << (estimator.predict(X) - y).squaredNorm() / X.cols() << std::endl;

    // Eigen::MatrixXd Xt = 5 * Eigen::MatrixXd::Random(2, 1000);
    // Eigen::MatrixXd yt = (0.5 * M_PI * Xt.colwise().norm()).array().cos() *
    //                      (-0.1 * M_PI * Xt.colwise().norm()).array().exp();

    // std::cout << (estimator.predict(Xt) - yt).squaredNorm() / Xt.cols() << std::endl;
    // std::cout << (yt).squaredNorm() / Xt.cols() << std::endl;
    // std::cout << estimator.predict(Xt) - yt << std::endl;

    // auto pred = estimator.predict(X);
    // std::cout << pred.transpose() << std::endl;
    // std::cout << (pred - y).norm() << std::endl;
    // std::cout << y.norm() << std::endl;
    // std::cout << estimator.predict(X) - y << std::endl;
    //     DSOVK(auto gloss, auto phi, auto feature_sampler,
 //          auto gamma, double nu,
 //          int T, int r, int d, int seed = 0) :

    return 0;
}