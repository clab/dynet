#include "dynet/tensor-eigen.h"
#include "dynet/nodes-conv2d.h"

#include <algorithm>
#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <array>

#include "dynet/functors.h"
#include "dynet/nodes-impl-macros.h"
#include "third_party/eigen_spatial_convolutions.h"
#include "third_party/eigen_backward_spatial_convolutions.h"

#if HAVE_CUDA
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#include "dynet/cudnn-ops.h"
#endif

using namespace std;

namespace dynet {

#ifndef __CUDACC__

string Conv2D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "conv2d(" << arg_names[0] << ", f=" << arg_names[1];
  if (arg_names.size() == 3)
    s << ", b=" << arg_names[2];
  s << ")";
  return s.str();
}

Dim Conv2D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 && xs.size() != 3) {
    ostringstream s; s << "Conv2D requires either two or three inputs: " << xs;
    throw std::invalid_argument(s.str());
  }
  if ((xs[0].ndims() != 2 && xs[0].ndims() != 3) || xs[1].ndims() != 4 ||
      xs[1][2] != xs[0][2]) {
    ostringstream s; s << "Bad input dimensions in Conv2D: " << xs;
    throw std::invalid_argument(s.str());
  }
  if (is_valid && (xs[0][0] < xs[1][0] || xs[0][1] < xs[1][1])) {
    ostringstream s; s << "Bad input dimensions in Conv2D: in VALID convolution, the filter size must not be greater than the feature map size" << xs;
    throw std::invalid_argument(s.str());
  }
  if (xs.size() == 3) { //has bias term
    if (xs[2][0] != xs[1][3] || xs[2].ndims() != 1) {
      ostringstream s; s << "Bad input dimensions in Conv2D: " << xs;
      throw std::invalid_argument(s.str());
    }
  }
  unsigned bs = xs[0].batch_elems();
  std::vector<long> output_shape(3);
  output_shape[2] = static_cast<long>(xs[1].d[3]);
  for (unsigned i = 0; i < 2; ++i) {
    float input_dim = static_cast<float>(xs[0].d[i]);
    float kernel_dim = static_cast<float>(xs[1].d[i]);
    float s = static_cast<float>(stride[i]);
    if (is_valid) {
      output_shape[i] = static_cast<long>(ceil((input_dim - kernel_dim + 1) / s));
    } else {
      output_shape[i] = static_cast<long>(ceil(input_dim / s));
    }
  }
  return Dim(output_shape, bs);
}

int Conv2D::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  Sig s(nt::conv2d);
  // Note that autobatching will only occur when inputs are of batch size one
  // TODO: remove this restriction, allowing for combining batched inputs
  if(dim.bd == 1) {
    s.add_dim(cg.nodes[args[0]]->dim); // the input
    s.add_node(args[1]); // the filter
    s.add_int(static_cast<int>(is_valid));
    s.add_int(stride[0]);
    s.add_int(stride[1]);
    return sm.get_idx(s);
  } else {
    return 0;
  }
}

std::vector<int> Conv2D::autobatch_concat(const ComputationGraph & cg) const {
  vector<int> ret(args.size(), 0);
  if (dim.bd == 1) { ret[0] = 1; }
  return ret;
}

// size_t Conv2D::aux_storage_size() const {
//   vector<unsigned> input_size(arity());
//   for (unsigned i = 0; i < arity(); ++i) {
//     input_size[i] = get_cg()->nodes[args[i]]->dim.size();
//   }
//   size_t nbytes = 0;
// #if HAVE_CUDNN
//   nbytes += CudnnConvOp::workspace_size_limit_bytes;
//   nbytes += 3 * input_size[0] * sizeof(float);
// #else
//   nbytes += sizeof(float) * (input_size[0] + input_size[1] +
//       dim.size() + std::max(input_size[0], input_size[1]));
// #endif
//   return nbytes;
//   // return 0;
// }
#endif

template<class MyDevice>
void Conv2D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2 || xs.size() == 3, "Failed dimension check in Conv2D::forward, at least 2 inputs");
  DYNET_ASSERT(fx.d.bd == xs[0]->d.bd, "Failed dimension check in Conv2D::forward, batchsize not match");
  DYNET_ASSERT(fx.d[2] == xs[1]->d[3], "Failed dimension check in Conv2D::forward, #channel not match");
  AlignedMemoryPool* scratch_allocator = default_device->pools[(int)DeviceMempool::SCS];
#ifdef __CUDACC__
#if HAVE_CUDNN
  if (cudnn_conv_op_ == NULL)
    cudnn_conv_op_ = new CudnnConvOp(stride, is_valid);
  cudnn_conv_op_->forward_impl(dev, xs, fx);
#else
  throw std::runtime_error("Conv2D::forward_dev_impl not supported without CUDNN");
#endif
#else
  Eigen::PaddingType padding_type = is_valid ? Eigen::PADDING_VALID : Eigen::PADDING_SAME;
  //void* CHWN_x_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
  void* CHWN_x_mem = scratch_allocator->allocate(xs[0]->d.size() * sizeof(float));
  Tensor CHWN_x = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_x_mem), xs[0]->device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> shuffles;
  shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
  tb<3>(CHWN_x).device(*dev.edevice) = tb<3>(*xs[0]).shuffle(shuffles);
  //void* NCHW_f_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
  void* NCHW_f_mem = scratch_allocator->allocate(xs[1]->d.size() * sizeof(float));
  Tensor NCHW_f = Tensor(Dim({xs[1]->d[3], xs[1]->d[2], xs[1]->d[0], xs[1]->d[1]}), static_cast<float*>(NCHW_f_mem), xs[1]->device, DeviceMempool::FXS);
  shuffles[0] = 3; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 1;
  t<4>(NCHW_f).device(*dev.edevice) = t<4>(*xs[1]).shuffle(shuffles);
  //void* CHWN_y_mem = aux_mem_pool.allocate(fx.d.size() * sizeof(float));
  void* CHWN_y_mem = scratch_allocator->allocate(fx.d.size() * sizeof(float));
  Tensor CHWN_y = Tensor(Dim({fx.d[2], fx.d[0], fx.d[1]}, fx.d.bd), static_cast<float*>(CHWN_y_mem), fx.device, DeviceMempool::FXS);
  tb<3>(CHWN_y).device(*dev.edevice) = Eigen::SpatialConvolution(tb<3>(CHWN_x), t<4>(NCHW_f), stride[0], stride[1], padding_type);
  shuffles[0] = 1; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 3;
  tb<3>(fx).device(*dev.edevice) = tb<3>(CHWN_y).shuffle(shuffles);
  if (xs.size() == 3) {
    Tensor bias = Tensor(Dim({fx.d[0], fx.d[1], fx.d.bd}, 1), static_cast<float*>(CHWN_x_mem), xs[2]->device, DeviceMempool::FXS);
    for (unsigned i = 0; i < fx.d[2]; ++i) {
      TensorTools::constant(bias, vec(*xs[2])(i));
      tb<3>(fx).chip<2>(i).device(*dev.edevice) += t<3>(bias);
    }
  }
#endif
  scratch_allocator->free();
}

template<class MyDevice>
void Conv2D::backward_dev_impl(const MyDevice & dev,
                         const vector<const Tensor*>& xs,
                         const Tensor& fx,
                         const Tensor& dEdf,
                         unsigned i,
                         Tensor& dEdxi) const {
  // don't check those already checked in forward_impl
  DYNET_ASSERT(dEdf.d == fx.d, "Failed dimension check in Conv2D::backward");
  DYNET_ASSERT(dEdxi.d == xs[i]->d, "Failed dimension check in Conv2D::backward");
  DYNET_ASSERT(i <= 2, "Failed dimension check in Conv2D::backward");
  AlignedMemoryPool* scratch_allocator = default_device->pools[(int)DeviceMempool::SCS];
#ifdef __CUDACC__
#if HAVE_CUDNN
  if (cudnn_conv_op_ == NULL)
    cudnn_conv_op_ = new CudnnConvOp(stride, is_valid);
  cudnn_conv_op_->backward_impl(dev, xs, fx, dEdf, i, dEdxi);
#else
  throw std::runtime_error("Conv2D::backward_dev_impl not supported without CUDNN");
#endif
#else
  //void* CHWN_dy_mem = aux_mem_pool.allocate(dEdf.d.size() * sizeof(float));
  void* CHWN_dy_mem = scratch_allocator->allocate(dEdf.d.size() * sizeof(float));
  Tensor CHWN_dy = Tensor(Dim({dEdf.d[2], dEdf.d[0], dEdf.d[1]}, dEdf.d.bd), static_cast<float*>(CHWN_dy_mem), dEdf.device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> shuffles;
  shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
  tb<3>(CHWN_dy).device(*dev.edevice) = tb<3>(dEdf).shuffle(shuffles);
  if (i == 0) { //backward w.r.t the input
    //void* NCHW_f_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
    void* NCHW_f_mem = scratch_allocator->allocate(xs[1]->d.size() * sizeof(float));
    Tensor NCHW_f = Tensor(Dim({xs[1]->d[3], xs[1]->d[2], xs[1]->d[0], xs[1]->d[1]}), static_cast<float*>(NCHW_f_mem), xs[1]->device, DeviceMempool::FXS);
    shuffles[0] = 3; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 1;
    t<4>(NCHW_f).device(*dev.edevice) = t<4>(*xs[1]).shuffle(shuffles);
    //void* CHWN_dEdxi_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
    void* CHWN_dEdxi_mem = scratch_allocator->allocate(xs[0]->d.size() * sizeof(float));
    Tensor CHWN_dEdxi = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    tb<3>(CHWN_dEdxi).device(*dev.edevice) = Eigen::SpatialConvolutionBackwardInput(t<4>(NCHW_f), tb<3>(CHWN_dy), xs[0]->d[0], xs[0]->d[1], stride[0], stride[1]);
    //void* HWCN_dEdxi_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
    void* HWCN_dEdxi_mem = scratch_allocator->allocate(xs[0]->d.size() * sizeof(float));
    Tensor HWCN_dEdxi = Tensor(xs[0]->d, static_cast<float*>(HWCN_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    shuffles[0] = 1; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 3;
    tb<3>(HWCN_dEdxi).device(*dev.edevice) = tb<3>(CHWN_dEdxi).shuffle(shuffles);
    tb<3>(dEdxi).device(*dev.edevice) += tb<3>(HWCN_dEdxi);
  } else if (i == 1) { //backward w.r.t the kernel
    //void* CHWN_x_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
    void* CHWN_x_mem = scratch_allocator->allocate(xs[0]->d.size() * sizeof(float));
    Tensor CHWN_x = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_x_mem), xs[0]->device, DeviceMempool::FXS);
    shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
    tb<3>(CHWN_x).device(*dev.edevice) = tb<3>(*xs[0]).shuffle(shuffles);
    //void* NCHW_dEdxi_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
    void* NCHW_dEdxi_mem = scratch_allocator->allocate(xs[1]->d.size() * sizeof(float));
    Tensor NCHW_dEdxi = Tensor(Dim({xs[1]->d[3], xs[1]->d[2], xs[1]->d[0], xs[1]->d[1]}), static_cast<float*>(NCHW_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    t<4>(NCHW_dEdxi).device(*dev.edevice) = Eigen::SpatialConvolutionBackwardKernel(tb<3>(CHWN_x), tb<3>(CHWN_dy), xs[1]->d[0], xs[1]->d[1], stride[0], stride[1], is_valid);
    //void* HWCN_dEdxi_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
    void* HWCN_dEdxi_mem = scratch_allocator->allocate(xs[1]->d.size() * sizeof(float));
    Tensor HWCN_dEdxi = Tensor(xs[1]->d, static_cast<float*>(HWCN_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    shuffles[0] = 2; shuffles[1] = 3; shuffles[2] = 1; shuffles[3] = 0;
    t<4>(HWCN_dEdxi).device(*dev.edevice) = t<4>(NCHW_dEdxi).shuffle(shuffles);
    t<4>(dEdxi).device(*dev.edevice) += t<4>(HWCN_dEdxi);
  } else { //backward w.r.t the bias
    Eigen::array<int, 3> red_axis = {0, 1, 3};
    t<1>(dEdxi).device(*dev.edevice) += tb<3>(dEdf).sum(red_axis);
  }
#endif
  scratch_allocator->free();
}
DYNET_NODE_INST_DEV_IMPL(Conv2D)

} // namespace dynet
