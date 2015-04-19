#ifndef CNN_TENSOR_H_
#define CNN_TENSOR_H_

#include "config.h"

#ifdef WITH_MINERVA_BACKEND
#  include "backends/minerva/tensor-minerva.h"
#endif

#ifdef WITH_THPP_BACKEND
#  include "backends/thpp/tensor.h"
#endif

#ifdef WITH_EIGEN_BACKEND
#  include "backends/eigen/tensor-eigen.h"
#endif

#endif
