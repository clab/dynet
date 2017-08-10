#ifndef DYNET_CUDNN_OPS_H
#define DYNET_CUDNN_OPS_H

#if HAVE_CUDNN
#include "dynet/dynet.h"
#include "dynet/cuda.h"

namespace dynet {

class CudnnConvOp {
 public:
  explicit CudnnConvOp() {}
  explicit CudnnConvOp(const std::vector<unsigned>& s, const bool padding_type);
  ~CudnnConvOp() noexcept(false);
  /* call this function before using the CudnnConvOp */
  void forward_impl(const Device_GPU & dev,
                    const std::vector<const Tensor*>& xs, Tensor& fx);
  void backward_impl(const Device_GPU & dev,
               const std::vector<const Tensor*>& xs,
               const Tensor& fx,
               const Tensor& dEdf,
               unsigned i,
               Tensor& dEdxi);
  static const size_t workspace_size_limit_bytes = 8 * 1024 * 1024;

 protected:
  std::vector<int> stride_;
  bool is_valid_;

  /* cuDNN resource */
  cudnnTensorDescriptor_t x_desc_, y_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_f_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_d_algo_;

  // cudnn workspace
  size_t workspace_fwd_size_;
  size_t workspace_bwd_data_size_;
  size_t workspace_bwd_filter_size_;
  void* fwd_workspace;
  void* bwd_filter_workspace;
  void* bwd_data_workspace;
};


class CudnnMaxPooling2DOp {
 public: 
  explicit CudnnMaxPooling2DOp() {}
  explicit CudnnMaxPooling2DOp(const std::vector<unsigned>& ksize,
                               const std::vector<unsigned>& stride,
                               const bool padding_type);
  ~CudnnMaxPooling2DOp() noexcept(false);
  void forward_impl(const Device_GPU & dev,
                    const std::vector<const Tensor*>& xs, Tensor& fx);
  void backward_impl(const Device_GPU & dev,
                const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi);

 protected:
  std::vector<int> ksize_;
  std::vector<int> stride_;
  bool is_valid_;

  /* cuDNN resource */
  cudnnTensorDescriptor_t x_desc_, y_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
};

} // namespace dynet

#endif
#endif
