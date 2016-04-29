#include "learning_rate.hpp"

using namespace Eigen;
using namespace std;

LearningRate::LearningRate(const learning_rate_t & learning_rate) :
    _learning_rate(learning_rate)
{

}

LearningRate::LearningRate(const LearningRate & learning_rate) :
    _learning_rate(learning_rate._learning_rate)
{

}
