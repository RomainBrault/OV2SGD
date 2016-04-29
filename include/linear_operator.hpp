#ifndef LINEAR_OPERATOR_HPP_INCLUDED
#define LINEAR_OPERATOR_HPP_INCLUDED

#include <utility>
#include <iostream>

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#include "Eigen/Dense"
#include "Eigen/Sparse"

template <typename LHS_T, typename RHS_T>
class DecomposableLinOp;

template <typename M, typename LHS_T, typename RHS_T>
class DecomposableLinOp_ProductReturnType;

namespace Eigen {
namespace internal {

    template <typename LHS_T, typename RHS_T>
    struct traits<DecomposableLinOp<LHS_T, RHS_T>> :
        Eigen::internal::traits<Eigen::SparseMatrix<double>>
    {

    };

    template <typename M, typename LHS_T, typename RHS_T>
    struct traits<DecomposableLinOp_ProductReturnType<M, LHS_T, RHS_T>>
    {
        // The equivalent plain objet type of the product.
        // This type is used if the product needs to be evaluated into a
        // temporary.
        using ReturnType = Eigen::Matrix<typename M::Scalar,
                                         Eigen::Dynamic,
                                         M::ColsAtCompileTime>;
    };

} // namespace internal
} // namespace Eigen

template <typename LHS_T, typename RHS_T>
class DecomposableLinOp :
    public Eigen::EigenBase<DecomposableLinOp<LHS_T, RHS_T>>
{
private:
    LHS_T _Lhs;
    RHS_T _Rhs;

public:
    // Expose some compile-time information to Eigen:
    using Scalar = double;
    using RealScalar = double;
    using Index = long long int;

    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        RowsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        MaxRowsAtCompileTime = Eigen::Dynamic
    };

    inline DecomposableLinOp(void) = default;

    inline DecomposableLinOp(long int rows_Lhs, long int cols_Lhs,
                             long int rows_Rhs, long int cols_Rhs) :
        _Lhs(rows_Lhs, cols_Lhs), _Rhs(rows_Rhs, cols_Rhs)
    {

    }

    template <typename M1, typename M2>
    inline DecomposableLinOp(const M1& Lhs, const M2& Rhs) :
        _Lhs(Lhs), _Rhs(Rhs)
    {

    }

    template <typename M1, typename M2>
    inline DecomposableLinOp(M1&& Lhs, M2&& Rhs) :
        _Lhs(std::forward<M1>(Lhs)), _Rhs(std::forward<M2>(Rhs))
    {

    }

    template <typename M1, typename M2>
    inline DecomposableLinOp(const M1& Lhs, M2&& Rhs) :
        _Lhs(Lhs), _Rhs(std::forward<M2>(Rhs))
    {

    }

    template <typename M1, typename M2>
    inline DecomposableLinOp(M1&& Lhs, const M2& Rhs) :
        _Lhs(std::forward<M1>(Lhs)), _Rhs(Rhs)
    {

    }

    inline DecomposableLinOp(const DecomposableLinOp & linop) :
        _Lhs(linop.get_Lhs()), _Rhs(linop.get_Rhs())
    {

    }

    inline DecomposableLinOp(DecomposableLinOp && linop) :
        _Lhs(std::move(linop.get_Lhs())), _Rhs(std::move(linop.get_Rhs()))
    {

    }

    inline auto set_Lhs(const Eigen::MatrixXd & Lhs)
        -> DecomposableLinOp &
    {
        _Lhs = Lhs;
        return *this;
    }

    inline auto set_Lhs(Eigen::MatrixXd && Lhs)
        -> DecomposableLinOp &
    {
        _Lhs = std::move(Lhs);
        return *this;
    }

    inline auto get_Lhs(void)
        -> LHS_T &
    {
        return _Lhs;
    }

    inline auto get_Lhs(void) const
        -> const LHS_T &
    {
        return _Lhs;
    }

    inline auto set_Rhs(const Eigen::MatrixXd & Rhs)
        -> DecomposableLinOp &
    {
        _Rhs = Rhs;
        return *this;
    }

    inline auto set_Rhs(Eigen::MatrixXd && Rhs)
        -> DecomposableLinOp &
    {
        _Rhs = std::move(Rhs);
        return *this;
    }

    inline auto get_Rhs(void)
        -> RHS_T &
    {
        return _Rhs;
    }

    inline auto get_Rhs(void) const
        -> const RHS_T &
    {
        return _Rhs;
    }

    inline auto rows() const
        -> Index
    {
        return _Lhs.rows() * _Rhs.rows();
    }

    inline auto cols() const
        -> Index
    {
        return _Lhs.cols() * _Rhs.cols();
    }

    inline auto resize(Index a_rows, Index a_cols)
        -> void
    {
        assert(a_rows == 0 && a_cols == 0 ||
               a_rows == rows() && a_cols == cols());
    }

    inline auto transpose(void)
        /* -> unspecified */
    {
        using LHS_T_T = decltype(get_Lhs().transpose());
        using RHS_T_T = decltype(get_Rhs().transpose());
        return DecomposableLinOp<LHS_T_T, RHS_T_T>(get_Lhs().transpose(),
                                                   get_Rhs().transpose());
    }

    inline auto leftCols(long int idx) const
        /* -> unspecified */
    {
        using LHS_T_T = decltype(get_Lhs().leftCols(idx));
        using RHS_T_T = decltype(get_Rhs());
        return DecomposableLinOp<LHS_T_T, RHS_T_T>(get_Lhs().leftCols(idx),
                                                   get_Rhs());
    }

    inline auto rightCols(long int idx) const
        /* -> unspecified */
    {
        using LHS_T_T = decltype(get_Lhs().rightCols(idx));
        using RHS_T_T = decltype(get_Rhs());
        return DecomposableLinOp<LHS_T_T, RHS_T_T>(get_Lhs().rightCols(idx),
                                                   get_Rhs());
    }

    template<typename M>
    inline auto operator*(const Eigen::MatrixBase<M>& x) const
        -> DecomposableLinOp_ProductReturnType<M, LHS_T, RHS_T>
    {
        return DecomposableLinOp_ProductReturnType<M, LHS_T, RHS_T>(
            *this, x.derived());
    }

    inline auto operator=(const DecomposableLinOp & linop)
        -> DecomposableLinOp &
    {
        set_Lhs(linop.get_Lhs());
        set_Rhs(linop.get_Rhs());
        return *this;
    }

    inline auto operator=(DecomposableLinOp && linop)
        -> DecomposableLinOp &
    {
        set_Lhs(std::move(linop.get_Lhs()));
        set_Rhs(std::move(linop.get_Rhs()));
        return *this;
    }

};

template <typename M, typename LHS_T, typename RHS_T>
class DecomposableLinOp_ProductReturnType
    : public Eigen::ReturnByValue<
        DecomposableLinOp_ProductReturnType<M, LHS_T, RHS_T>>
{
public:
    using Index = long long int;

    // The ctor store references to the matrix and right-hand-side object
    // (usually a vector).
    DecomposableLinOp_ProductReturnType(
        const DecomposableLinOp<LHS_T, RHS_T>& op, const M& rhs)
        : m_matrix(op), m_rhs(rhs)
    {

    }

    inline auto rows() const
        -> Index
    {
        return m_matrix.rows();
    }

    inline auto cols() const
        -> Index
    {
        return m_rhs.cols();
    }

    // This function is automatically called by Eigen.
    // It must evaluate the product of matrix * rhs into y.
    template<typename Dest>
    void evalTo(Dest& y) const
    {
        Eigen::Map<Eigen::Matrix<double,
                                 Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
            x_rhp(const_cast<typename M::Scalar*>(m_rhs.data()),
                  m_matrix.get_Lhs().cols(), m_matrix.get_Rhs().cols());
        Eigen::Map<Eigen::Matrix<double,
                                 Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
            y_rhp(y.data(),
                  m_matrix.get_Lhs().rows(), m_matrix.get_Rhs().rows());

        y_rhp = m_matrix.get_Lhs() * x_rhp * m_matrix.get_Rhs().transpose();
    }

protected:
    const DecomposableLinOp<LHS_T, RHS_T>& m_matrix;
    typename M::Nested m_rhs;
};

#endif // LINEAR_OPERATOR_HPP_INCLUDED