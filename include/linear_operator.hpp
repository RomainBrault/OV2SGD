#ifndef LINEAR_OPERATOR_HPP_INCLUDED
#define LINEAR_OPERATOR_HPP_INCLUDED

#include <utility>
#include <iostream>

#ifdef RELEASE
#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#endif
#include "Eigen/Dense"
#include "Eigen/Sparse"

template <typename LHS_T, typename RHS_T>
class DecomposableLinOp;

namespace Eigen {
namespace internal {

    // DecomposableLinOp looks-like a SparseMatrix, so let's inherits its
    // traits:
    template <typename LHS_T, typename RHS_T>
    struct traits<DecomposableLinOp<LHS_T, RHS_T>> :
        public Eigen::internal::traits<Eigen::SparseMatrix<double>>
    {

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
    mutable Eigen::MatrixXd _temp;

public:
    // Expose some compile-time information to Eigen:
    using Scalar = double;
    using RealScalar = double;
    using StorageIndex = long int;

    static constexpr bool IsRowMajor = false;

    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        RowsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        MaxRowsAtCompileTime = Eigen::Dynamic
    };

    inline DecomposableLinOp(void) = default;

    inline DecomposableLinOp(long int rows_Lhs, long int cols_Lhs,
                             long int rows_Rhs, long int cols_Rhs) :
        _Lhs(rows_Lhs, cols_Lhs), _Rhs(rows_Rhs, cols_Rhs),
        _temp(cols_Lhs, cols_Rhs)
    {

    }

    template <typename M1, typename M2>
    inline DecomposableLinOp(const M1& Lhs, const M2& Rhs) :
        _Lhs(Lhs), _Rhs(Rhs),
        _temp(get_Lhs().cols(), get_Rhs().cols())
    {

    }

    template <typename M1, typename M2>
    inline DecomposableLinOp(M1&& Lhs, M2&& Rhs) :
        _Lhs(std::forward<M1>(Lhs)), _Rhs(std::forward<M2>(Rhs)),
        _temp(get_Lhs().cols(), get_Rhs().cols())
    {

    }

    template <typename M1, typename M2>
    inline DecomposableLinOp(const M1& Lhs, M2&& Rhs) :
        _Lhs(Lhs), _Rhs(std::forward<M2>(Rhs)),
        _temp(get_Lhs().cols(), get_Rhs().cols())
    {

    }

    template <typename M1, typename M2>
    inline DecomposableLinOp(M1&& Lhs, const M2& Rhs) :
        _Lhs(std::forward<M1>(Lhs)), _Rhs(Rhs),
        _temp(get_Lhs().cols(), get_Rhs().cols())
    {

    }

    inline DecomposableLinOp(const DecomposableLinOp & linop) :
        _Lhs(linop.get_Lhs()), _Rhs(linop.get_Rhs()),
        _temp(linop._temp)
    {

    }

    inline DecomposableLinOp(DecomposableLinOp && linop) :
        _Lhs(std::move(linop.get_Lhs())), _Rhs(std::move(linop.get_Rhs())),
        _temp(std::move(linop._temp))
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

    inline auto get_temp(void) const
        -> Eigen::MatrixXd &
    {
        return _temp;
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
        -> StorageIndex
    {
        return _Lhs.rows() * _Rhs.rows();
    }

    inline auto cols() const
        -> StorageIndex
    {
        return _Lhs.cols() * _Rhs.cols();
    }

    inline auto resize(StorageIndex a_rows, StorageIndex a_cols)
        -> void
    {
        assert(a_rows == 0 && a_cols == 0 ||
               a_rows == rows() && a_cols == cols());
    }

    inline auto transpose(void)
        -> DecomposableLinOp<decltype(this->get_Lhs().transpose()),
                             decltype(this->get_Rhs().transpose())>
    {
        using LHS_T_T = decltype(get_Lhs().transpose());
        using RHS_T_T = decltype(get_Rhs().transpose());
        return DecomposableLinOp<LHS_T_T, RHS_T_T>(get_Lhs().transpose(),
                                                   get_Rhs().transpose());
    }

    inline auto leftCols(long int idx) const
        -> DecomposableLinOp<decltype(this->get_Lhs().leftCols(idx)),
                             decltype(this->get_Rhs())>
    {
        using LHS_T_T = decltype(get_Lhs().leftCols(idx));
        using RHS_T_T = decltype(get_Rhs());
        return DecomposableLinOp<LHS_T_T, RHS_T_T>(get_Lhs().leftCols(idx),
                                                   get_Rhs());
    }

    inline auto rightCols(long int idx) const
        -> DecomposableLinOp<decltype(this->get_Lhs().rightCols(idx)),
                             decltype(this->get_Rhs())>
    {
        using LHS_T_T = decltype(get_Lhs().rightCols(idx));
        using RHS_T_T = decltype(get_Rhs());
        return DecomposableLinOp<LHS_T_T, RHS_T_T>(get_Lhs().rightCols(idx),
                                                   get_Rhs());
    }

    template<typename Rhs>
    auto operator*(const Eigen::MatrixBase<Rhs>& x) const
        -> Eigen::Product<DecomposableLinOp<LHS_T, RHS_T>, Rhs,
                          Eigen::AliasFreeProduct>
    {
        return Eigen::Product<DecomposableLinOp<LHS_T, RHS_T>, Rhs,
                              Eigen::AliasFreeProduct>(*this, x.derived());
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

namespace Eigen {
namespace internal {

    template <typename Rhs, typename LHS_T, typename RHS_T>
    struct generic_product_impl<DecomposableLinOp<LHS_T, RHS_T>, Rhs,
                                SparseShape, DenseShape, GemvProduct> :
        generic_product_impl_base<DecomposableLinOp<LHS_T, RHS_T>, Rhs,
                                  generic_product_impl<
                                      DecomposableLinOp<LHS_T, RHS_T>,Rhs>>
    {
        using Scalar =
            typename Product<DecomposableLinOp<LHS_T, RHS_T>,Rhs>::Scalar;

        template <typename Dest>
        static auto scaleAndAddTo(Dest& dst,
                                  const DecomposableLinOp<LHS_T, RHS_T>& lhs,
                                  const Rhs& rhs, const Scalar& alpha)
            -> void
        {
            Eigen::Map<Eigen::Matrix<double,
                                     Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                x_rhp(const_cast<typename Rhs::Scalar*>(rhs.data()),
                      lhs.get_Lhs().cols(), lhs.get_Rhs().cols());
            Eigen::Map<Eigen::Matrix<double,
                                     Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                y_rhp(dst.data(), lhs.get_Lhs().rows(), lhs.get_Rhs().rows());

            lhs.get_temp().noalias() = x_rhp * lhs.get_Rhs().transpose();
            y_rhp.noalias() = lhs.get_Lhs() * lhs.get_temp();
            y_rhp *= alpha;
        }
    };
} // namespace Eigen
} // namespace internal

#endif // LINEAR_OPERATOR_HPP_INCLUDED