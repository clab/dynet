#ifndef DYNET_CUDA_MATRIX_MULTIPLY_H__
#define DYNET_CUDA_MATRIX_MULTIPLY_H__

#include "dynet/tensor.h"
#include "dynet/tensor-eigen.h"
#include "dynet/devices.h"
#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

#ifdef __CUDACC__

#include "dynet/cuda.h"

namespace dynet {

inline void MatrixMultiply(const Device_GPU & dev, const Tensor& l, const Tensor& r, Tensor& y, const float* acc_scalar) {
  CUDA_CHECK(cudaSetDevice(dev.cuda_device_id));
  if(l.d.bd == 1 && r.d.bd == y.d.bd) {
    // If the left side has one batch, multiply by columns
    // [x, z, b] = [x, y] * [y, z, b]
    // -> [x, z*b] = [x, y], [y, z*b]
    CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
          y.d.rows(), y.d.cols() * y.d.batch_elems(), l.d.cols(),
          dev.kSCALAR_ONE,
          l.v, l.d.rows(),
          r.v, r.d.rows(),
          acc_scalar, y.v, y.d.rows()));
  } else {
    CUBLAS_CHECK(cublasSgemmStridedBatched(dev.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
          y.d.rows(), y.d.cols(), l.d.cols(),
          dev.kSCALAR_ONE,
          l.v, l.d.rows(), (l.d.bd > 1 ? l.d.batch_size() : 0),
          r.v, r.d.rows(), (r.d.bd > 1 ? r.d.batch_size() : 0),
          acc_scalar, y.v, y.d.rows(), y.d.batch_size(),
          y.d.bd));
  }
}

}

#else

namespace dynet {

inline void MatrixMultiply(const Device_CPU & dev, const Tensor& l, const Tensor& r, Tensor& y, const float* acc_scalar) {

  tbvec(y).device(*dev.edevice) = *acc_scalar * tbvec(y);

  if(l.d.bd == 1 && r.d.bd == y.d.bd) {

      // If the left side has one batch, multiply by columns
      // [x, z, b] = [x, y] * [y, z, b]
      // -> [x, z*b] = [x, y], [y, z*b]
      colbatch_matrix(y).noalias() += mat(l) * colbatch_matrix(r);

  } else {
    #ifdef __INTEL_MKL__
      // grp_cout is 1 as matrices in a batch have equal shape
      const MKL_INT grp_count = 1;
      const float *a_array[y.d.bd], *b_array[y.d.bd];
      float *c_array[y.d.bd];
      MKL_INT m[grp_count] = {l.d.rows()};
      MKL_INT k[grp_count] = {r.d.rows()};
      MKL_INT n[grp_count] = {r.d.cols()};
      MKL_INT lda[grp_count] = {l.d.rows()};
      MKL_INT ldb[grp_count] = {r.d.rows()};
      MKL_INT ldc[grp_count] = {y.d.rows()};
      MKL_INT size_per_grp[grp_count] = {y.d.bd};
      CBLAS_TRANSPOSE transA[grp_count] = {CblasNoTrans};
      CBLAS_TRANSPOSE transB[grp_count] = {CblasNoTrans};

      for (uint i = 0; i < y.d.bd; ++i) {
        a_array[i] = l.v + i*(l.d.bd > 1 ? l.d.batch_size() : 0);
        b_array[i] = r.v + i*(r.d.bd > 1 ? r.d.batch_size() : 0);
        c_array[i] = y.v + i*(y.d.bd > 1 ? y.d.batch_size() : 0);
      }
      cblas_sgemm_batch (CblasColMajor, transA, transB,
            m, n, k,
            dev.kSCALAR_ONE,
            a_array, lda,
            b_array, ldb,
            acc_scalar, c_array, ldc,
            grp_count, size_per_grp);
    #else
      // Otherwise, loop over the batches
      for(unsigned b = 0; b < y.d.bd; ++b)
        batch_matrix(y, b).noalias() += batch_matrix(l, b) * batch_matrix(r, b);
    #endif
  }
}

}

#endif

#ifdef __CUDACC__
inline void MatrixTranspMultiplyAcc(const dynet::Device_GPU & dev, const dynet::Tensor& l, const dynet::Tensor& r, dynet::Tensor& y) {
  // computes l^T * r
  int max_b = std::max(l.d.bd, r.d.bd);
  // Do a single multiply if l has one batch
  if(l.d.bd == 1 && y.d.bd == r.d.bd) {
    CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          y.d.rows(), y.d.cols()*y.d.batch_elems(), l.d.rows(),
          dev.kSCALAR_ONE,
          l.v, l.d.rows(),
          r.v, r.d.rows(),
          dev.kSCALAR_ONE, y.v, y.d.rows()));
  } else {
    CUBLAS_CHECK(cublasSgemmStridedBatched(dev.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          y.d.rows(), y.d.cols(), l.d.rows(),
          dev.kSCALAR_ONE,
          l.v, l.d.rows(), (l.d.bd > 1 ? l.d.batch_size() : 0),
          r.v, r.d.rows(), (r.d.bd > 1 ? r.d.batch_size() : 0),
          dev.kSCALAR_ONE, y.v, y.d.rows(), y.d.batch_size(),
          max_b));
  }
}

# else
inline void MatrixTranspMultiplyAcc(const dynet::Device_CPU & dev, const dynet::Tensor& l, const dynet::Tensor& r, dynet::Tensor& y) {
  // computes l^T * r
  int max_b = std::max(l.d.bd, r.d.bd);
  if(l.d.bd == 1 && y.d.bd == r.d.bd) {
    colbatch_matrix(y).noalias() += mat(l).transpose() * colbatch_matrix(r);
  } else {
    #ifdef __INTEL_MKL__
      // grp_cout is 1 as matrices in a batch have equal shape
      const MKL_INT grp_count = 1;
      const float *a_array[max_b-1], *b_array[max_b-1];
      float *c_array[max_b-1];
      MKL_INT m[grp_count] = {l.d.cols()};
      MKL_INT k[grp_count] = {r.d.rows()};
      MKL_INT n[grp_count] = {r.d.cols()};
      MKL_INT lda[grp_count] = {l.d.rows()};
      MKL_INT ldb[grp_count] = {r.d.rows()};
      MKL_INT ldc[grp_count] = {y.d.rows()};
      MKL_INT size_per_grp[grp_count] = {max_b};
      CBLAS_TRANSPOSE transA[grp_count] = {CblasNoTrans};
      CBLAS_TRANSPOSE transB[grp_count] = {CblasTrans};

      for (uint i = 0; i < max_b; ++i) {
        a_array[i] = l.v + i*(l.d.bd > 1 ? l.d.batch_size() : 0);
        b_array[i] = r.v + i*(r.d.bd > 1 ? r.d.batch_size() : 0);
        c_array[i] = y.v + i*(y.d.bd > 1 ? y.d.batch_size() : 0);
      }
      cblas_sgemm_batch (CblasColMajor, transA, transB,
            m, n, k,
            dev.kSCALAR_ONE,
            a_array, lda,
            b_array, ldb,
            dev.kSCALAR_ONE, c_array, ldc,
            grp_count, size_per_grp);
    #else
      for(int b = 0; b < max_b; ++b)
        batch_matrix(y, b).noalias() += batch_matrix(l, b).transpose() * batch_matrix(r, b);
    #endif
  }
}
#endif

#ifdef __CUDACC__
inline void MatrixMultiplyTranspAcc(const dynet::Device_GPU & dev, const dynet::Tensor& l, const dynet::Tensor& r, dynet::Tensor& y) {
  int max_b = std::max(l.d.bd, r.d.bd);
  if(y.d.bd == 1 && (l.d.bd == r.d.bd)) {
    CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
          y.d.rows(), y.d.cols(), l.d.cols() * l.d.batch_elems(),
          dev.kSCALAR_ONE,
          l.v, l.d.rows(),
          r.v, r.d.rows(),
          dev.kSCALAR_ONE, y.v, y.d.rows()));
  } else {
    CUBLAS_CHECK(cublasSgemmStridedBatched(dev.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
          y.d.rows(), y.d.cols(), l.d.cols(),
          dev.kSCALAR_ONE,
          l.v, l.d.rows(), (l.d.bd > 1 ? l.d.batch_size() : 0),
          r.v, r.d.rows(), (r.d.bd > 1 ? r.d.batch_size() : 0),
          dev.kSCALAR_ONE, y.v, y.d.rows(), y.d.batch_size(),
          max_b));
  }
}

# else
inline void MatrixMultiplyTranspAcc(const dynet::Device_CPU & dev, const dynet::Tensor& l, const dynet::Tensor& r, dynet::Tensor& y) {
  int max_b = std::max(l.d.bd, r.d.bd);
  if(y.d.bd == 1 && (l.d.bd == r.d.bd)) {
    mat(y).noalias() += colbatch_matrix(l) * colbatch_matrix(r).transpose();
  } else {
    #ifdef __INTEL_MKL__
      // grp_cout is 1 as matrices in a batch have equal shape
      const MKL_INT grp_count = 1;
      const float *a_array[max_b-1], *b_array[max_b-1];
      float *c_array[max_b-1];
      MKL_INT m[grp_count] = {l.d.rows()};
      MKL_INT k[grp_count] = {r.d.cols()};
      MKL_INT n[grp_count] = {r.d.rows()};
      MKL_INT lda[grp_count] = {l.d.rows()};
      MKL_INT ldb[grp_count] = {r.d.rows()};
      MKL_INT ldc[grp_count] = {y.d.rows()};
      MKL_INT size_per_grp[grp_count] = {max_b};
      CBLAS_TRANSPOSE transA[grp_count] = {CblasNoTrans};
      CBLAS_TRANSPOSE transB[grp_count] = {CblasTrans};

      for (uint i = 0; i < max_b; ++i) {
        a_array[i] = l.v + i*(l.d.bd > 1 ? l.d.batch_size() : 0);
        b_array[i] = r.v + i*(r.d.bd > 1 ? r.d.batch_size() : 0);
        c_array[i] = y.v + i*(y.d.bd > 1 ? y.d.batch_size() : 0);
      }
      cblas_sgemm_batch (CblasColMajor, transA, transB,
            m, n, k,
            dev.kSCALAR_ONE,
            a_array, lda,
            b_array, ldb,
            dev.kSCALAR_ONE, c_array, ldc,
            grp_count, size_per_grp);
    #else
      for(int b = 0; b < max_b; ++b)
        batch_matrix(y, b).noalias() += batch_matrix(l, b) * batch_matrix(r, b).transpose();
    #endif
  }
}
#endif


#endif
