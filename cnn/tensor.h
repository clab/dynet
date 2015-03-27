#ifndef CNN_TENSOR_H_
#define CNN_TENSOR_H_

#include "config.h"
#ifdef HAVE_MINERVA_H
//#  include "backends/minerva/tensor-minerva.h"
#  include "backends/eigen/tensor-eigen.h"
#else
#  include "backends/eigen/tensor-eigen.h"
#endif

#endif
