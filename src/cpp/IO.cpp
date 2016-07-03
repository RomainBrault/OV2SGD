#include <vector>

#include "IO.hpp"

using namespace Eigen;
using namespace std;

namespace libsvm {

auto read(const string & filename,
                 long int n, long int d, long int start)
    -> pair<MatrixXd, MatrixXd>
{
    MatrixXd X(d, n);
    VectorXd y(n);

    libsvm::read(filename, X, y, start);
    return make_pair(move(X), move(y));
}

auto read(istream & stream,
                 long int n, long int d, long int start)
    -> pair<MatrixXd, MatrixXd>
{
    MatrixXd X(d, n);
    VectorXd y(n);

    libsvm::read(stream, X, y, start);
    return make_pair(move(X), move(y));
}

} // namespace libsvm