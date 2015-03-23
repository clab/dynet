#ifndef SAXE_INIT_H_
#define SAXE_INIT_H_

#include "cnn/tensor.h"

namespace cnn {

// returns a dim x dim matrix
Tensor OrthonormalRandom(unsigned dim, real g);

}

#endif
