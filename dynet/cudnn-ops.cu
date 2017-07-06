#if HAVE_CUDNN
#include <iostream>
#include <vector>
#include <algorithm>

#include "dynet/dynet.h"
#include "dynet/cudnn-ops.h"

#define CUDNN_VERSION_MIN(major, minor, patch) \
        (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

namespace dynet {

CudnnConvOp::CudnnConvOp(const std::vector<unsigned>& s, const bool padding_type) {
  stride_.resize(s.size());
  for (unsigned i = 0; i < stride_.size(); ++i) {
    stride_[i] = static_cast<int>(s[i]);
  }
  is_valid_ = padding_type;
  fwd_workspace = NULL;
  bwd_filter_workspace = NULL;
  bwd_data_workspace = NULL;
  workspace_fwd_size_ = 0;
  workspace_bwd_data_size_ = 0;
  workspace_bwd_filter_size_ = 0;
  mempool_ = NULL;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
}

CudnnConvOp::~CudnnConvOp() noexcept(false) {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void CudnnConvOp::forward_impl(const Device_GPU& dev, const std::vector<const Tensor*>& xs, Tensor& fx) {
  if (mempool_ == NULL)
    throw std::runtime_error("dynet::CudnnConvOp::mempool_ not set");

  const Tensor* x = xs[0]; 
  const Tensor* filter = xs[1];
  Tensor* y = &fx;
  unsigned XN = x->d.bd;
  unsigned XC = x->d[2];
  unsigned XH = x->d[0];
  unsigned XW = x->d[1];
  unsigned FYC = filter->d[3];
  unsigned FXC = filter->d[2];
  unsigned FH = filter->d[0];
  unsigned FW = filter->d[1];
  unsigned YN = fx.d.bd;
  unsigned YC = fx.d[2];
  unsigned YH = fx.d[0];
  unsigned YW = fx.d[1];

  int pad_h = 0, pad_w = 0;
  bool h_odd = false, w_odd = false;
  // infer pad_h, pad_w
  // Total padding on rows and cols is
  // pad_h = (YH - 1) * stride[0] + FH - XH
  // pad_w = (YW - 1) * stride[1] + FW - XW
  // We pad pad_h/2 on the left and pad_h - pad_h/2 on the right, pad_w/2 on the top
  // and pad_w - pad_w/2 on the bottom.  When pad_h or pad_w is odd, this means
  // we pad more on the right and bottom than on the top and left.
  if (!is_valid_) {
    pad_h = std::max<int>(0, (YH - 1) * stride_[0] + FH - XH);
    pad_w = std::max<int>(0, (YW - 1) * stride_[1] + FW - XW);
    h_odd = (pad_h % 2 != 0);
    w_odd = (pad_w % 2 != 0);
    if (h_odd || w_odd) { // then we need to pad one row/col on the bottom/right
      unsigned new_XH = XH + h_odd;
      unsigned new_XW = XW + w_odd;
      void* temp = mempool_->allocate(sizeof(float) * new_XW * new_XH * XC * XN);
      Tensor padded_x = Tensor(Dim({new_XH, new_XW, XC}, XN), static_cast<float*>(temp), xs[0]->device, DeviceMempool::FXS);
      Eigen::array<std::pair<int, int>, 4> paddings;
      paddings[0] = std::make_pair(0, static_cast<int>(h_odd));
      paddings[1] = std::make_pair(0, static_cast<int>(w_odd));
      paddings[2] = std::make_pair(0, 0);
      paddings[3] = std::make_pair(0, 0);
      padded_x.tb<3>().device(*dev.edevice) = xs[0]->tb<3>().pad(paddings);
      // re-point x to the padded input
      XH = new_XH;
      XW = new_XW;
      x = &padded_x;
    }
  }

  // set cudnn descriptors
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_, 
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              XN, XC, XW, XH));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc_, 
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              YN, YC, YW, YH));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, 
              DataTypeToCudnnType<float>::value, CUDNN_TENSOR_NCHW,
              FYC, FXC, FW, FH));
#else
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(filter_desc_, 
              DataTypeToCudnnType<float>::value, CUDNN_TENSOR_NCHW,
              FYC, FXC, FW, FH));
#endif
#if CUDNN_VERSION_MIN(6, 0, 0)
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_,
              pad_w/2, pad_h/2, stride_[1], stride_[0], 1, 1,
              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#else
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_,
              pad_w/2, pad_h/2, stride_[1], stride_[0], 1, 1,
              CUDNN_CROSS_CORRELATION));
#endif
  if (xs.size() == 3) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc_,
                CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
                1, FYC, 1, 1));
  }

  // TODO(Hao Zhang): there should be an autotune function to determine
  // the best convolution algorithm to use.
  // However, as DyNet changes CG for every sample (or every iteration),
  // This autotune function seems to be unnecessary.
  // Note: this following computations are *NON-DETERMINISTIC*
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(dev.cudnnHandle,
              x_desc_, filter_desc_, conv_desc_, y_desc_,
              CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, workspace_size_limit_bytes,
              &fwd_algo_));
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(dev.cudnnHandle,
              x_desc_, filter_desc_, conv_desc_, y_desc_,
              fwd_algo_, &workspace_fwd_size_));
  fwd_workspace = mempool_->allocate(workspace_fwd_size_);
  float alpha = 1.f, beta = 0.f;
  CUDNN_CHECK(cudnnConvolutionForward(dev.cudnnHandle,
              &alpha, x_desc_, x->v, filter_desc_, filter->v,
              conv_desc_, fwd_algo_, fwd_workspace, workspace_fwd_size_,
              &beta, y_desc_, y->v));
  if (xs.size() == 3) {
    CUDNN_CHECK(cudnnAddTensor(dev.cudnnHandle, &alpha, 
                bias_desc_, xs[2]->v, &alpha, y_desc_, y->v));
  }
}

// We don't suppose backward_impl depends on the exeuction of forward_impl
// i.e. backward_impl can be called independently
void CudnnConvOp::backward_impl(const Device_GPU & dev, 
             const std::vector<const Tensor*>& xs,
             const Tensor& fx,
             const Tensor& dEdf,
             unsigned i,
             Tensor& dEdxi) {
  if (mempool_ == NULL)
    throw std::runtime_error("dynet::CudnnConvOp::mempool_ not set");

  const Tensor* x = xs[0]; 
  const Tensor* filter = xs[1];
  const Tensor* dy = &dEdf;
  void* dxi = NULL;

  unsigned XN = x->d.bd;
  unsigned XC = x->d[2];
  unsigned XH = x->d[0];
  unsigned XW = x->d[1];
  unsigned FYC = filter->d[3];
  unsigned FXC = filter->d[2];
  unsigned FH = filter->d[0];
  unsigned FW = filter->d[1];
  unsigned YN = fx.d.bd;
  unsigned YC = fx.d[2];
  unsigned YH = fx.d[0];
  unsigned YW = fx.d[1];

  // create padded input if necessary
  int pad_h = 0, pad_w = 0;
  bool h_odd = false, w_odd = false;
  if (!is_valid_) {
    pad_h = std::max<int>(0, (YH - 1) * stride_[0] + FH - XH);
    pad_w = std::max<int>(0, (YW - 1) * stride_[1] + FW - XW);
    h_odd = (pad_h % 2 != 0);
    w_odd = (pad_w % 2 != 0);
    if (h_odd || w_odd) {
      unsigned new_XH = XH + h_odd;
      unsigned new_XW = XW + w_odd;
      void* temp = mempool_->allocate(sizeof(float) * new_XW * new_XH * XC * XN);
      Tensor padded_x = Tensor(Dim({new_XH, new_XW, XC}, XN), static_cast<float*>(temp), xs[0]->device, DeviceMempool::FXS);
      Eigen::array<std::pair<int, int>, 4> paddings;
      paddings[0] = std::make_pair(0, static_cast<int>(h_odd));
      paddings[1] = std::make_pair(0, static_cast<int>(w_odd));
      paddings[2] = std::make_pair(0, 0);
      paddings[3] = std::make_pair(0, 0);
      padded_x.tb<3>().device(*dev.edevice) = xs[0]->tb<3>().pad(paddings);
      // re-point x to the padded input
      XH = new_XH;
      XW = new_XW;
      x = &padded_x;
    }
  }

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_,
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              XN, XC, XW, XH));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc_,
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              YN, YC, YW, YH));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, 
              DataTypeToCudnnType<float>::value, CUDNN_TENSOR_NCHW,
              FYC, FXC, FW, FH));
#else
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(filter_desc_, 
              DataTypeToCudnnType<float>::value, CUDNN_TENSOR_NCHW,
              FYC, FXC, FW, FH));
#endif
#if CUDNN_VERSION_MIN(6, 0, 0)
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_,
              pad_w/2, pad_h/2, stride_[1], stride_[0], 1, 1,
              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#else
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_,
              pad_w/2, pad_h/2, stride_[1], stride_[0], 1, 1,
              CUDNN_CROSS_CORRELATION));
#endif
  if (i == 2) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc_,
                CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
                1, FYC, 1, 1));
  }
  float alpha = 1.f, beta = 0.f;
  switch(i) {
    case 0: { // grad w.r.t. feature maps
      dxi = mempool_->allocate(sizeof(float) * XH * XW * XC * XN);
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(dev.cudnnHandle,
                  filter_desc_, y_desc_, conv_desc_, x_desc_,
                  CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                  workspace_size_limit_bytes, &bwd_d_algo_));
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(dev.cudnnHandle,
                  filter_desc_, y_desc_, conv_desc_, x_desc_,
                  bwd_d_algo_, &workspace_bwd_data_size_));
      bwd_data_workspace = mempool_->allocate(workspace_bwd_data_size_);
      CUDNN_CHECK(cudnnConvolutionBackwardData(dev.cudnnHandle,
                  &alpha, filter_desc_, filter->v, y_desc_, dy->v,
                  conv_desc_, bwd_d_algo_, bwd_data_workspace, workspace_bwd_data_size_,
                  &beta, x_desc_, dxi));
      Tensor padded_dx = Tensor(Dim({XH, XW, XC}, XN), static_cast<float*>(dxi), xs[0]->device, DeviceMempool::FXS);
      //std::cout << padded_dx.d << " " << dEdxi.d << std::endl;
      Eigen::array<int, 4> offsets = {0, 0, 0, 0};
      Eigen::array<int, 4> extents = {static_cast<int>(XH - h_odd), static_cast<int>(XW - w_odd), static_cast<int>(XC), static_cast<int>(XN)};
      dEdxi.tb<3>().device(*dev.edevice) += padded_dx.tb<3>().slice(offsets, extents);
    } break;
    case 1: {// grad w.r.t. filters
      dxi = mempool_->allocate(sizeof(float) * FYC * FXC * FW * FH);
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(dev.cudnnHandle,
                  x_desc_, y_desc_, conv_desc_, filter_desc_,
                  CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                  workspace_size_limit_bytes, &bwd_f_algo_));
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(dev.cudnnHandle,
                  x_desc_, y_desc_, conv_desc_, filter_desc_,
                  bwd_f_algo_, &workspace_bwd_filter_size_));
      bwd_filter_workspace = mempool_->allocate(workspace_bwd_filter_size_);
      CUDNN_CHECK(cudnnConvolutionBackwardFilter(dev.cudnnHandle,
                  &alpha, x_desc_, x->v, y_desc_, dy->v,
                  conv_desc_, bwd_f_algo_, bwd_filter_workspace, workspace_bwd_filter_size_,
                  &beta, filter_desc_, dxi));
      //accumlate the gradient
      Tensor dxi_tensor = Tensor(Dim({FH, FW, FXC}, FYC), static_cast<float*>(dxi), xs[1]->device, DeviceMempool::FXS);
      dEdxi.t<4>().device(*dev.edevice) += dxi_tensor.t<4>();
    } break;
    case 2: {// grad w.r.t. bias
      dxi = mempool_->allocate(sizeof(float) * FYC);
      CUDNN_CHECK(cudnnConvolutionBackwardBias(dev.cudnnHandle,
                  &alpha, y_desc_, dy->v,
                  &beta, bias_desc_, dxi));
      CUDNN_CHECK(cudnnAddTensor(dev.cudnnHandle, &alpha,
                  bias_desc_, dxi, &alpha, bias_desc_, dEdxi.v));
    } break;
    default:
      throw std::runtime_error("dynet::CudnnConvOp::backward_impl, conv2d have at most 3 inputs");
  }
}

CudnnMaxPooling2DOp::CudnnMaxPooling2DOp(const std::vector<unsigned>& ksize,
                                         const std::vector<unsigned>& stride,
                                         const bool padding_type) {
  ksize_.resize(ksize.size());
  stride_.resize(stride.size());
  for (unsigned i = 0; i < ksize.size(); ++i) {
    ksize_[i] = static_cast<int>(ksize[i]);
    stride_[i] = static_cast<int>(stride[i]);
  }
  is_valid_ = padding_type;
  mempool_ = NULL;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc_));
}

CudnnMaxPooling2DOp::~CudnnMaxPooling2DOp() noexcept(false) {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_));
  CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_desc_));
}

void CudnnMaxPooling2DOp::forward_impl(const Device_GPU & dev,
                                       const std::vector<const Tensor*>& xs,
                                       Tensor& fx) {
  const Tensor* x = xs[0];
  Tensor* y = &fx;

  unsigned XN = x->d.bd;
  unsigned XC = x->d[2];
  unsigned XH = x->d[0];
  unsigned XW = x->d[1];
  unsigned YN = fx.d.bd;
  unsigned YC = fx.d[2];
  unsigned YH = fx.d[0];
  unsigned YW = fx.d[1];

  // infer pad_h, pad_w
  int pad_h = 0, pad_w = 0;
  if (!is_valid_) {
    pad_h = std::max<int>(0, (YH - 1) * stride_[0] + ksize_[0] - XH) / 2;
    pad_w = std::max<int>(0, (YW - 1) * stride_[1] + ksize_[1] - XW) / 2;
  }

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_,
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              XN, XC, XW, XH));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc_,
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              YN, YC, YW, YH));
  #if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc_,
                CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                ksize_[1], ksize_[0], pad_w, pad_h, stride_[1], stride_[0]));
  #else
    CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(pooling_desc_,
                CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                ksize_[1], ksize_[0], pad_w, pad_h, stride_[1], stride_[0]));
  #endif
  float alpha = 1.f, beta = 0.f;
  CUDNN_CHECK(cudnnPoolingForward(dev.cudnnHandle, pooling_desc_, 
              &alpha, x_desc_, x->v,
              &beta, y_desc_, y->v));
}

void CudnnMaxPooling2DOp::backward_impl(const Device_GPU & dev,
              const std::vector<const Tensor*>& xs,
              const Tensor& fx,
              const Tensor& dEdf,
              unsigned i,
              Tensor& dEdxi) {
  if (mempool_ == NULL)
    throw std::runtime_error("dynet::CudnnMaxPooling2DOp::mempool_ not set");
  const Tensor* x = xs[0]; 
  const Tensor* y = &fx; 
  const Tensor* dy = &dEdf;
  void* dxi = NULL;

  unsigned XN = x->d.bd;
  unsigned XC = x->d[2];
  unsigned XH = x->d[0];
  unsigned XW = x->d[1];
  unsigned YN = fx.d.bd;
  unsigned YC = fx.d[2];
  unsigned YH = fx.d[0];
  unsigned YW = fx.d[1];

  // infer pad_h, pad_w
  int pad_h = 0, pad_w = 0;
  if (!is_valid_) {
    pad_h = std::max<int>(0, (YH - 1) * stride_[0] + ksize_[0] - XH) / 2;
    pad_w = std::max<int>(0, (YW - 1) * stride_[1] + ksize_[1] - XW) / 2;
  }

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_,
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              XN, XC, XW, XH));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc_,
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              YN, YC, YW, YH));
  #if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc_,
                CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                ksize_[1], ksize_[0], pad_w, pad_h, stride_[1], stride_[0]));
  #else
    CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(pooling_desc_,
                CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                ksize_[1], ksize_[0], pad_w, pad_h, stride_[1], stride_[0]));
  #endif

  // here we could reuse the descriptor we created for forward, because
  // they share the same size
  float alpha = 1.f, beta = 0.f;
  dxi = mempool_->allocate(sizeof(float) * XN * XC * XH * XW);
  CUDNN_CHECK(cudnnPoolingBackward(dev.cudnnHandle, pooling_desc_,
              &alpha, y_desc_, y->v, y_desc_, dy->v,
              x_desc_, x->v, &beta, x_desc_, dxi));
  CUDNN_CHECK(cudnnAddTensor(dev.cudnnHandle, &alpha,
              x_desc_, dxi, &alpha, x_desc_, dEdxi.v));
}

} // namespace dynet

#endif
