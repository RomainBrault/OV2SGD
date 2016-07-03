#ifndef IO_HPP_INCLUDED
#define IO_HPP_INCLUDED

#include <istream>
#include <ostream>
#include <fstream>
#include <iostream>
#include <exception>
#include <utility>

#ifdef RELEASE
#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#endif
#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace libsvm {

template <typename M1, typename M2>
auto read(std::istream & stream, M1 & X, M2 & y, long int start)
    -> void
{
    X.setZero();
    for (long int i = 0; i < start; ++i) {
        stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    for (long int i = 0; i < X.rows(); ++i) {
        stream >> y(i);
        while (stream.peek() != '\n' && !stream.eof()) {
            long int pos;
            stream >> pos;
            stream.ignore(1, ':');
            stream >> X(pos, i);
        }
    }
}

template <typename M1, typename M2>
auto read(const std::string & filename, M1 & X, M2 & y, long int start)
    -> void
{
    std::ifstream stream;
    stream.open(filename, std::ifstream::in);
    if (!stream.is_open()) {
        throw std::runtime_error("Cannot open file: \"" + filename + "\".");
    }
    libsvm::read(stream, X, y, start);
}

auto read(const std::string & filename,
                 long int n, long int d, long int start = 0)
    -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd>;

auto read(std::istream & stream,
                 long int n, long int d, long int start = 0)
    -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd>;

template <typename T>
inline auto read_sparse(std::istream & stream,
                        long int n, long int d, long int start = 0)
    -> std::pair<Eigen::SparseMatrix<T>, Eigen::SparseMatrix<T>>
{
    Eigen::Matrix<T, 1, Eigen::Dynamic> y(n);
    std::vector<Eigen::Triplet<T>> tripletList;
    tripletList.reserve(n);

    for (long int i = 0; i < start; ++i) {
        stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    for (long int i = 0; i < n; ++i) {
        stream >> y(i);
        while (stream.peek() != '\n' && !stream.eof()) {
            long int pos;
            T value;
            stream >> pos;
            stream.ignore(1, ':');
            stream >> value;
            tripletList.push_back(Eigen::Triplet<T>(pos, i, value));
        }
    }
    Eigen::SparseMatrix<T> X(d, n);
    X.setFromTriplets(tripletList.begin(), tripletList.end());
    X.makeCompressed();
    Eigen::SparseMatrix<T> ys(y.maxCoeff() + 1, n);
    ys.reserve(Eigen::VectorXi::Constant(ys.cols(), 1));
    for (long int i = 0; i < n; ++i) {
        ys.insert(y(i), i) = 1;
    }
    ys.makeCompressed();
    return std::make_pair(std::move(X), std::move(ys));
}

template <>
inline auto read_sparse<unsigned char>(std::istream & stream,
                                       long int n, long int d, long int start)
    -> std::pair<Eigen::SparseMatrix<unsigned char>,
                 Eigen::SparseMatrix<unsigned char>>
{
    Eigen::Matrix<unsigned char, 1, Eigen::Dynamic> y(n);
    std::vector<Eigen::Triplet<unsigned char>> tripletList;
    tripletList.reserve(n);

    for (long int i = 0; i < start; ++i) {
        stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    for (long int i = 0; i < n; ++i) {
        unsigned int value;
        stream >> value;
        y(i) = value;
        while (stream.peek() != '\n' && !stream.eof()) {
            long int pos;
            unsigned int value;
            stream >> pos;
            stream.ignore(1, ':');
            stream >> value;
            tripletList.push_back(
                Eigen::Triplet<unsigned char>(
                    pos, i, static_cast<unsigned char>(value)));
        }
    }
    Eigen::SparseMatrix<unsigned char> X(d, n);
    X.setFromTriplets(tripletList.begin(), tripletList.end());
    X.makeCompressed();
    Eigen::SparseMatrix<unsigned char> ys(y.maxCoeff() + 1, n);
    ys.reserve(Eigen::VectorXi::Constant(ys.cols(), 1));
    for (long int i = 0; i < n; ++i) {
        ys.insert(y(i), i) = 1;
    }
    ys.makeCompressed();
    return std::make_pair(std::move(X), std::move(ys));
}

template <typename T>
auto read_sparse(const std::string & filename,
                 long int n, long int d, long int start = 0)
    -> std::pair<Eigen::SparseMatrix<T>, Eigen::SparseMatrix<T>>
{
    std::ifstream stream;
    stream.open(filename, std::ifstream::in);
    if (!stream.is_open()) {
        throw std::runtime_error("Cannot open file: \"" + filename + "\".");
    }
    return libsvm::read_sparse<T>(stream, n, d, start);
}

} // namespace libsvm


namespace binary {

template <typename M>
auto write(const std::string & filename, const M& matrix)
    -> void
{
    std::ofstream out(filename,
                      std::ios::out | std::ios::binary | std::ios::trunc);

    if (out.is_open()) {
        typename M::Index rows = matrix.rows(),
                          cols = matrix.cols();
        out.write(reinterpret_cast<const char*>(&rows),
                  sizeof(typename M::Index));
        out.write(reinterpret_cast<const char*>(&cols),
                  sizeof(typename M::Index));
        out.write(reinterpret_cast<const char*>(matrix.data()),
                  rows * cols * sizeof(typename M::Scalar));
    }
}

template <typename M>
auto read(const std::string & filename, M& matrix)
    -> void
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);

    if (in.is_open()) {
        typename M::Index rows = 0,
                          cols = 0;
        in.read(reinterpret_cast<char *>(&rows), sizeof(typename M::Index));
        in.read(reinterpret_cast<char *>(&cols), sizeof(typename M::Index));

        matrix.resize(rows, cols);
        in.read(reinterpret_cast<char*>(matrix.data()),
                rows * cols * sizeof(typename M::Scalar));
    }
}

template <typename T, typename U>
auto read(const std::string & filename)
    -> Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic>
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m;
    read(filename, m);
    return m.template cast<U>();
}

template <typename T, int whatever, typename IND>
auto write_sparse(const std::string & filename,
                  Eigen::SparseMatrix<T, whatever, IND>& m)
    -> void
{
    std::vector<Eigen::Triplet<T>> res;

    std::ofstream out(filename,
                      std::ios::binary | std::ios::out | std::ios::trunc);

    using iter_t = typename
        Eigen::SparseMatrix<T, whatever, IND>::InnerIterator;
    if (out.is_open()) {
        m.makeCompressed();
        const IND xyn[3] {static_cast<IND>(m.rows()),
                          static_cast<IND>(m.cols()),
                          static_cast<IND>(m.nonZeros())};
        out.write(reinterpret_cast<const char*>(xyn), 3 * sizeof(IND));
        for (IND k = 0; k < m.outerSize(); ++k) {
            for (iter_t it(m, k); it; ++it) {
                IND rc[2] {static_cast<IND>(it.row()),
                           static_cast<IND>(it.col())};
                out.write(reinterpret_cast<const char*>(&rc), 2 * sizeof(IND));
                T v = it.value();
                out.write(reinterpret_cast<const char*>(&v), sizeof(T));
            }
        }
    }
}

template <typename T, typename U, int whatever, typename IND>
auto read_sparse(const std::string & filename,
                 Eigen::SparseMatrix<U, whatever, IND>& m)
    -> void
{
    std::ifstream in(filename, std::ios::binary | std::ios::in);

    if(in.is_open()) {
        IND xyn[3] {0, 0, 0};
        in.read(reinterpret_cast<char*>(xyn), 3 * sizeof(IND));
        m.resize(xyn[0], xyn[1]);
        std::vector<Eigen::Triplet<T>> trips(xyn[2]);
        IND rc[2];
        T   v {0};
        for (IND k=0; k < trips.size(); ++k) {
            in.read(reinterpret_cast<char*>(rc), 2 * sizeof(IND));
            in.read(reinterpret_cast<char*>(&v), sizeof(T));
            trips[k] = {rc[0], rc[1], v};
        }
        m.setFromTriplets(trips.begin(), trips.end());
        m.makeCompressed();
    }
}

template <typename T, typename U>
auto read_sparse(const std::string & filename)
    -> Eigen::SparseMatrix<U>
{
    Eigen::SparseMatrix<U> m;
    binary::read_sparse<T, U>(filename, m);
    return m;
}

} // namespace binary

// template <typename T, int whatever, typename IND>
// auto write_batch_sparse(const std::string & filename,
//                         Eigen::SparseMatrix<T, whatever, IND>& m,
//                         long int n_batch)
//     -> void
// {
//     m.makeCompressed();

//     long int batch_size = m.cols() / n_batch;
//     for (long int t = 0; t < n_batch; ++t) {
//         std::string batch_filename(filename + "_batch_" +
//                                    std::to_string(t) + ".bin");
//         binary::write_sparse(batch_filename,
//                              m.middleCols(t * batch_size, batch_size));
//     }
// }

// template<class Matrix>
// void read_binary(const char* filename, Matrix& matrix){
//     std::ifstream in(filename,ios::in | std::ios::binary);
//     typename Matrix::Index rows=0, cols=0;
//     in.read((char*) (&rows),sizeof(typename Matrix::Index));
//     in.read((char*) (&cols),sizeof(typename Matrix::Index));
//     matrix.resize(rows, cols);
//     in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
//     in.close();
// }

#endif