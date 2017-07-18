#ifndef DYNET_CPU_MATRIX_MULTIPLY_H__
#define DYNET_CPU_MATRIX_MULTIPLY_H__

#include "dynet/tensor.h"
#include "dynet/devices.h"

namespace dynet {

inline void CPUMatrixMultiply(const Device_CPU & dev, const Tensor& l, const Tensor& r, Tensor& y, const float* acc_scalar) {

  y.tbvec().device(*dev.edevice) = *acc_scalar * y.tbvec();

  if(l.d.bd == 1 && r.d.bd == y.d.bd) {

      // If the left side has one batch, multiply by columns
      // [x, z, b] = [x, y] * [y, z, b]
      // -> [x, z*b] = [x, y], [y, z*b]
      y.colbatch_matrix().noalias() += *l * r.colbatch_matrix();

  } else {
    // Otherwise, loop over the batches
    DYNET_ASSERT(r.d.bd != 1 || r.d.bd != l.d.bd,
                 "Number of batch elements in matrix multiply must match, but got: " << r.d.bd << ", " << l.d.bd);

    for(unsigned b = 0; b < y.d.bd; ++b)
      y.batch_matrix(b).noalias() += l.batch_matrix(b) * r.batch_matrix(b);

  }
}

}

#endif
