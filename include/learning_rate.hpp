#ifndef LEARNING_RATE_HPP_INCLUDED
#define LEARNING_RATE_HPP_INCLUDED

#include <functional>

#ifdef RELEASE
#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#endif
#include "Eigen/Dense"

class LearningRate
{
public:
    using learning_rate_t = std::function<double(long int)>;

private:
    learning_rate_t _learning_rate;

public:
    LearningRate(const learning_rate_t & learning_rate);

    LearningRate(const LearningRate & learning_rate);

    inline auto operator ()(long int i)
        -> double
    {
        return _learning_rate(i);
    }

};

inline auto InverseScaling(double gamma0, double gamma1, double alpha)
    -> LearningRate
{
    return LearningRate([gamma0, gamma1, alpha](long int i) -> double
    {
        return gamma0 * std::pow(static_cast<double>(1 + gamma1 * i), alpha);
    });
}

inline auto AverageScaling(long int d, long int n)
    -> LearningRate
{
    return LearningRate([d, n](long int i) -> double
    {
        return 1. / std::max<double>(std::max<double>(1, i - d), i - n);
    });
}

#endif // LEARNING_RATE_HPP_INCLUDED