#ifndef CNN_TENSOR_H_
#define CNN_TENSOR_H_

#include "config.h"
#include "cnn/dim.h"

#ifdef WITH_EIGEN_BACKEND
#  include "backends/eigen/init.h"
#  include "backends/eigen/tensor.h"
#else
#  error "Don't know any backend but Eigen"
#endif

#endif
