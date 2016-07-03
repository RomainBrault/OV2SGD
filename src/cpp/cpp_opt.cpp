#include "opt.hpp"

using namespace Eigen;
using namespace std;
using namespace sitmo;

DSOVK::DSOVK(const loss_t & gloss,
             const feature_map_t & feature_map,
             const gamma_t & gamma,
             long int p,
             double nu1, double nu2, long int T,
             long int batch, long int block, long int T_cap, double cond) :
    _gloss(gloss),
    _phi(feature_map),
    _gamma(gamma), _p(p),
    _nu1(nu1), _nu2(nu2), _T(T), _T_cap(T_cap), _d(0),
    _batch(batch), _block(block), _cond(cond),
    _coefs(2 * feature_map.r() * get_T_cap() * block)
{
    initParallel();
}

TSOVK::TSOVK(const loss_t & gloss,
             const feature_map_t & feature_map,
             const gamma_t & gamma,
             long int p,
             double nu1, double nu2, long int T,
             long int batch, long int block, long int T_cap, double cond) :
    _gloss(gloss),
    _phi(feature_map),
    _gamma(gamma), _p(p),
    _nu1(nu1), _nu2(nu2), _T(T), _T_cap(T_cap), _d(0),
    _batch(batch), _block(block), _cond(cond),
    _coefs(2 * get_T_cap() * block)
{
    initParallel();
}

DivSOVK::DivSOVK(const loss_t & gloss,
                 const feature_map_t & feature_map,
                 const gamma_t & gamma,
                 long int p,
                 double nu1, double nu2, long int T,
                 long int batch, long int block, long int T_cap, double cond) :
    _gloss(gloss),
    _phi(feature_map),
    _gamma(gamma), _p(p),
    _nu1(nu1), _nu2(nu2), _T(T), _T_cap(T_cap), _d(0),
    _batch(batch), _block(block), _cond(cond),
    _coefs(2 * (p - 1) * get_T_cap() * block)
{
    initParallel();
}
