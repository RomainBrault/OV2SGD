// #include "aopt.hpp"

// using namespace Eigen;
// using namespace std;
// using namespace sitmo;

// ADSOVK::ADSOVK(const loss_t & gloss,
//                const feature_map_t & feature_map,
//                const gamma_t & gamma,
//                const mu_t & mu,
//                double nu, long int T,
//                long int p,
//                long int batch, long int block, long int T_cap) :
//     _gloss(gloss),
//     _phi(feature_map),
//     _gamma(gamma), _mu(mu),
//     _nu(nu), _T(T), _T_cap(T_cap), _d(0), _p(p),
//     _batch(batch), _block(block),
//     _coefs(2 * feature_map.r() * get_T_cap() * block),
//     _coefs_avg(2 * feature_map.r() * get_T_cap() * block)
// {
//     initParallel();
// }
