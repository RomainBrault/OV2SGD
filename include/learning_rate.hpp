#ifndef LEARNING_RATE_HPP_INCLUDED
#define LEARNING_RATE_HPP_INCLUDED

#include <functional>

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
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

inline auto InverseScaling(double gamma0, double gamma1)
    -> LearningRate
{
    return LearningRate([gamma0, gamma1](long int i) -> double
    {
        return gamma0 / (1. + gamma1 * i);
    });
}

#endif // LEARNING_RATE_HPP_INCLUDED