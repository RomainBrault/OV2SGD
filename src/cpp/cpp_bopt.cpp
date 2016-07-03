// #include "bopt.hpp"

// using namespace Eigen;
// using namespace std;
// using namespace sitmo;

// BDSOVK::BDSOVK(const loss_t & gloss,
//                const feature_map_t & feature_map,
//                const gamma_t & gamma, double nu1, double nu2, long int T,
//                long int p,
//                long int batch, long int block, long int T_cap, double cond) :
//     _gloss(gloss),
//     _phi(feature_map),
//     _gamma(gamma),
//     _nu1(nu1), _nu2(nu2), _T(T), _T_cap(T_cap), _d(0), _p(p),
//     _batch(batch), _block(block), _cond(cond),
//     _coefs(2 * feature_map.r() * get_T_cap() * block),
//     _B(get_r(), get_p())
// {
//     initParallel();
// }
