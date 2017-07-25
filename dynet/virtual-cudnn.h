#ifndef DYNET_VIRTUAL_CUDNN_H
#define DYNET_VIRTUAL_CUDNN_H
#if HAVE_CUDNN

#include "dynet/cuda.h"
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
  return "Unknown cudnn status";
}



namespace cudnn {


inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}


inline void destroyTensorDesc(cudnnTensorDescriptor_t desc) {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
}

inline void setTensor4dDescriptor(cudnnTensorDescriptor_t desc, int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc,
              CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
              n, c, h, w));
}


inline void createFilterDesc(cudnnFilterDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
}


inline void setFilterDesc(cudnnFilterDescriptor_t* desc, int n, int c, int h, int w) {
  #if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, 
              CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
              n, c, h, w));
#else
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(filter_desc_, 
              DataTypeToCudnnType<float>::value, CUDNN_TENSOR_NCHW,
              FYC, FXC, FW, FH));
#endif
}


inline void destroyFilterDesc(cudnnFilterDescriptor_t desc) {
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc));
}


inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}


inline void destroyConvolutionDesc(cudnnConvolutionDescriptor_t conv) {
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv));
}


inline void setConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
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


inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc) {
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
}

inline void setPoolingDesc(cudnnPoolingDescriptor_t* pool_desc, 
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

inline void destroyPoolingDesc(cudnnPoolingDescriptor_t poolingDesc) {
  CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
}

//convolution operations
inline void getConvolutionForwardAlgorithm(cudnnHandle_t handle, const cudnnTensorDescriptor_t x_desc, const cudnnFilterDescriptor_t filter_desc, const cudnnConvolutionDescriptor_t conv_desc, const cudnnTensorDescriptor_t y_desc, size_t max_bytes, cudnnConvolutionFwdAlgo_t* fw_alg) {
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
              x_desc, filter_desc, conv_desc, y_desc,
              CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, max_bytes,
              fw_alg));
}

inline void getConvolutionForwardWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t x_desc, const cudnnFilterDescriptor_t filter_desc, const cudnnConvolutionDescriptor_t conv_desc, const cudnnTensorDescriptor_t y_desc, cudnnConvolutionFwdAlgo_t fw_alg, size_t *workspace_fwd_size) {
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
              x_desc, filter_desc, conv_desc, y_desc,
              fw_alg, workspace_fwd_size));
}

inline void convolutionForward(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t x_desc, const void* x_v, const cudnnFilterDescriptor_t filter_desc, const void* filter_v, const cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionFwdAlgo_t fw_alg, void* fwd_workspace, size_t workspace_fwd_size, const void* beta, const cudnnTensorDescriptor_t y_desc, void* y_v) {
  CUDNN_CHECK(cudnnConvolutionForward(handle,
              alpha, x_desc, x_v, filter_desc, filter_v,
              conv_desc, fw_alg, fwd_workspace, workspace_fwd_size,
              beta, y_desc, y_v));
}


inline void addTensor(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t aDesc, const void* A, const void* beta, const cudnnTensorDescriptor_t cDesc, void* C) {
  CUDNN_CHECK(cudnnAddTensor(handle, alpha,
              aDesc, A, beta, cDesc, C));
}

inline void getConvolutionBackwardDataAlgorithm(cudnnHandle_t handle, const cudnnFilterDescriptor_t filter_desc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t conv_desc, const cudnnTensorDescriptor_t dxDesc, size_t max_bytes, cudnnConvolutionBwdDataAlgo_t* bw_alg) {
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle,
              filter_desc, dyDesc, conv_desc, dxDesc,
              CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
              max_bytes, bw_alg));
}


inline void getConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle, const cudnnFilterDescriptor_t filter_desc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t conv_desc, const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t bw_alg, size_t *sizeInBytes) {
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
              filter_desc, dyDesc, conv_desc, dxDesc,
              bw_alg, sizeInBytes));
}

inline void convolutionBackwardData(cudnnHandle_t handle, const void* alpha, const cudnnFilterDescriptor_t filter_desc, const void* filter_v, const cudnnTensorDescriptor_t dy_desc, const void* dy_v, const cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionBwdDataAlgo_t bw_alg, void* bw_workspace, size_t workspace_bw_size, const void* beta, const cudnnTensorDescriptor_t dx_desc, void* dx_v) {
  CUDNN_CHECK(cudnnConvolutionBackwardData(handle,
              alpha, filter_desc, filter_v, dy_desc, dy_v,
              conv_desc, bw_alg, bw_workspace, workspace_bw_size,
              beta, dx_desc, dx_v));
}


inline void getConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc, const cudnnConvolutionDescriptor_t conv_desc, const cudnnFilterDescriptor_t filter_desc, size_t max_bytes, cudnnConvolutionBwdFilterAlgo_t* bw_alg) {
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
              xDesc, yDesc, conv_desc, filter_desc,
              CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
              max_bytes, bw_alg));
}


inline void getConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t conv_desc, const cudnnFilterDescriptor_t filter_desc, cudnnConvolutionBwdFilterAlgo_t bw_alg, size_t *sizeInBytes) {
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
              xDesc, dyDesc, conv_desc, filter_desc,
              bw_alg, sizeInBytes));
}


inline void convolutionBackwardFilter(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t x_desc, const void* x_v , const cudnnTensorDescriptor_t dy_desc, const void* dy_v, const cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionBwdFilterAlgo_t bw_alg, void* bw_workspace, size_t workspace_bw_size, const void* beta, const cudnnFilterDescriptor_t filter_desc, void* filter_v) {
  CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle,
              alpha, x_desc, x_v, dy_desc, dy_v,
              conv_desc, bw_alg, bw_workspace, workspace_bw_size,
              beta, filter_desc, filter_v));
}



inline void convolutionBackwardBias(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t dyDesc, const void* dy, const void* beta, const cudnnTensorDescriptor_t dbDesc, void* db) {
  CUDNN_CHECK(cudnnConvolutionBackwardBias(handle,
              alpha, dyDesc, dy,
              beta, dbDesc, db));
}


inline void poolingForward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void* alpha, const cudnnTensorDescriptor_t x_desc, const void* x_v, const void* beta, const cudnnTensorDescriptor_t y_desc, void* y_v) {
  CUDNN_CHECK(cudnnPoolingForward(handle, poolingDesc,
              alpha, x_desc, x_v,
              beta, y_desc, y_v));
}

inline void poolingBackward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void* alpha, const cudnnTensorDescriptor_t yDesc, const void* y, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnTensorDescriptor_t xDesc, const void* xData, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx) {
  CUDNN_CHECK(cudnnPoolingBackward(handle, poolingDesc,
              alpha, yDesc, y, dyDesc, dy,
              xDesc, xData, beta, dxDesc, dx));
}
}
#endif
#endif
