#ifndef DYNET_RAND_H
#define DYNET_RAND_H

#include "dynet/tensor.h"

namespace dynet {

  /**
   * \ingroup random
   * \brief This is a helper function to sample uniformly in \f$[0,1]\f$
   * \return \f$x\sim\mathcal U([0,1])\f$
   */
  real rand01();
  /**
   * \ingroup random
   * \brief This is a helper function to sample uniformly in \f$\{0,\dots,n-1\}\f$
   *
   * \param n Upper bound (excluded)
   * \return \f$x\sim\mathcal U(\{0,\dots,n-1\})\f$
   */
  int rand0n(int n);
  /**
   * \ingroup random
   * \brief This is a helper function to sample from a normalized gaussian distribution
   *
   * \return \f$x\sim\mathcal N(0,1)\f$
   */
  real rand_normal();
  /**
   * \ingroup random
   * \brief This returns a new random seed.
   */
  int draw_random_seed();

} // namespace dynet

#endif
