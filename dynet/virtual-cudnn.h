#ifndef DYNET_VIRTUAL_CUDNN_H
#define DYNET_VIRTUAL_CUDNN_H
#if HAVE_CUDNN

#include <cudnn.h>
#include "dynet/dynet.h"

#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
#if CUDNN_VERSION_MIN(6, 0, 0)
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#endif
  }
  return "Unknown cuDNN status";
}

namespace cudnn {

inline void createTensorDescriptor(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

inline void destroyTensorDescriptor(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(*desc));
}

inline void setTensor4dDescriptor(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc,
              CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
              n, c, h, w));
}

inline void createFilterDescriptor(cudnnFilterDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
}

inline void setFilter4dDescriptor(cudnnFilterDescriptor_t* desc,
    int n, int c, int h, int w) {
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, 
              CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
              n, c, h, w));
#else
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(*desc,
              CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
              n, c, h, w));
#endif
}

inline void destroyFilterDescriptor(cudnnFilterDescriptor_t* desc) {
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(*desc));
}

inline void createConvolutionDescriptor(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

inline void destroyConvolutionDescriptor(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(*conv));
}

inline void setConvolution2dDescriptor(cudnnConvolutionDescriptor_t* conv,
    int pad_h, int pad_w, int stride_h, int stride_w) {
#if CUDNN_VERSION_MIN(6, 0, 0)
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
      pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT));
#else
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
      pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
#endif
}

inline void createPoolingDescriptor(cudnnPoolingDescriptor_t* pool_desc) {
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
}

inline void setPooling2dDescriptor(cudnnPoolingDescriptor_t* pool_desc,
                           cudnnPoolingMode_t mode, int h, int w, int pad_h,
                           int pad_w, int stride_h, int stride_w) {
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, mode,
        CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
#else
  CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(*pool_desc, mode,
        CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
#endif
}

inline void destroyPoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc) {
  CUDNN_CHECK(cudnnDestroyPoolingDescriptor(*poolingDesc));
}

}
#endif //end HAVE_CUDNN
#endif
