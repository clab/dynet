#include "dynet/tensor-eigen.h"
#include "dynet/nodes-maxpooling2d.h"

#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <array>

#include "dynet/functors.h"
#include "dynet/nodes-impl-macros.h"
#include "third_party/eigen_pooling.h"

#if HAVE_CUDA
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#endif

using namespace std;

namespace dynet {

#ifndef __CUDACC__

string MaxPooling2D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "maxpooling2d(" << arg_names[0] << ")";
  return s.str();
}

Dim MaxPooling2D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1) {
    ostringstream s; s << "MaxPooling2D requires exactly one input: " << xs;
    throw std::invalid_argument(s.str());
  }
  if (xs[0].ndims() != 2 && xs[0].ndims() != 3) {
    ostringstream s; s << "Bad input dimensions in MaxPooling2D, expected 2 or 3 dimensions: " << xs;
    throw std::invalid_argument(s.str());
  }
  if (is_valid && (xs[0].d[0] < ksize[0] || xs[0].d[1] < ksize[1])) {
    ostringstream s; s << "Bad input dimensions in MaxPooling2D: \
        in VALID mode, the kernel size cannot be greater than the feature map size" << xs;
    throw std::invalid_argument(s.str());
  }
  unsigned bs = xs[0].batch_elems();
  std::vector<long> output_shape(xs[0].ndims());
  if(xs[0].ndims() == 3)
    output_shape[2] = static_cast<long>(xs[0][2]);
  for (unsigned i = 0; i < 2; ++i) {
    float input_dim = static_cast<float>(xs[0][i]);
    float kernel_dim = static_cast<float>(ksize[i]);
    float s = static_cast<float>(stride[i]);
    if (is_valid) {
      output_shape[i] = static_cast<long>(ceil((input_dim - kernel_dim + 1) / s));
    } else {
      output_shape[i] = static_cast<long>(ceil(input_dim / s));
    }
  }
  return Dim(output_shape, bs);
}

#endif

template<class MyDevice>
void MaxPooling2D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in MaxPooling2D::forward, exactly one input");
  DYNET_ASSERT(fx.d.bd == xs[0]->d.bd, "Failed dimension check in MaxPooling2D::forward, batchsize not match");
  DYNET_ASSERT(fx.d[2] == xs[0]->d[2], "Failed dimension check in MaxPooling2D::forward, #channel not match");
  AlignedMemoryPool* scratch_allocator = default_device->pools[(int)DeviceMempool::SCS];
#ifdef __CUDACC__
#if HAVE_CUDNN
  if (cudnn_maxpool_op_ == NULL)
    cudnn_maxpool_op_ = new CudnnMaxPooling2DOp(ksize, stride, is_valid);
  cudnn_maxpool_op_->forward_impl(dev, xs, fx);
#else
  throw std::runtime_error("MaxPooling2D::forward_dev_impl not supported without CUDNN");
#endif
#else
  Eigen::PaddingType padding_type = is_valid ? Eigen::PADDING_VALID : Eigen::PADDING_SAME;
  // convert x from HWCN to CHWN
  void* CHWN_x_mem = scratch_allocator->allocate(xs[0]->d.size() * sizeof(float));
  Tensor CHWN_x = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_x_mem), xs[0]->device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> shuffles;
  shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
  tb<3>(CHWN_x).device(*dev.edevice) = tb<3>(*xs[0]).shuffle(shuffles);
  // allocate temp memory and compute
  void* CHWN_y_mem = scratch_allocator->allocate(fx.d.size() * sizeof(float));
  Tensor CHWN_y = Tensor(Dim({fx.d[2], fx.d[0], fx.d[1]}, fx.d.bd), static_cast<float*>(CHWN_y_mem), fx.device, DeviceMempool::FXS);
  tb<3>(CHWN_y).device(*dev.edevice) = Eigen::SpatialMaxPooling(tb<3>(CHWN_x), ksize[0], ksize[1], stride[0], stride[1], padding_type);
  // convert y from CHWN to HWCN
  shuffles[0] = 1; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 3;
  tb<3>(fx).device(*dev.edevice) = tb<3>(CHWN_y).shuffle(shuffles);
#endif
  scratch_allocator->free();
}

template<class MyDevice>
void MaxPooling2D::backward_dev_impl(const MyDevice & dev,
                         const vector<const Tensor*>& xs,
                         const Tensor& fx,
                         const Tensor& dEdf,
                         unsigned i,
                         Tensor& dEdxi) const {
  DYNET_ASSERT(dEdf.d == fx.d, "Failed dimension check in MaxPooling2D::backward");
  DYNET_ASSERT(dEdxi.d == xs[i]->d, "Failed dimension check in MaxPooling2D::backward");
  DYNET_ASSERT(i == 0, "Failed dimension check in MaxPooling2D::backward: i must be 0");
#ifdef __CUDACC__
#if HAVE_CUDNN
  if (cudnn_maxpool_op_ == NULL)
    cudnn_maxpool_op_ = new CudnnMaxPooling2DOp(ksize, stride, is_valid);
  cudnn_maxpool_op_->backward_impl(dev, xs, fx, dEdf, i, dEdxi);
#else
  throw std::runtime_error("MaxPooling2D::backward_dev_impl not supported without CUDNN");
#endif
#else
  int pad_along_height = ((fx.d[0] - 1) * stride[0] +
                  ksize[0] - xs[0]->d[0]);
  int pad_along_width = ((fx.d[1] - 1) * stride[1] +
                 ksize[1] - xs[0]->d[1]);
  int pad_top = is_valid ? 0 : pad_along_height / 2;
  int pad_left = is_valid ? 0 : pad_along_width / 2;
  for (unsigned b = 0; b < fx.d.bd; ++b) {
    for (unsigned i = 0; i < fx.d[0]; ++i) {
      for (unsigned j = 0; j < fx.d[1]; ++j) {
        for (unsigned ch = 0; ch < fx.d[2]; ++ch) {
          int max_r = 0, max_c = 0;
          float max_val;
          bool is_feasible = false;
          for (unsigned r = 0; r < ksize[0]; ++r) {
            for (unsigned c = 0; c < ksize[1]; ++c) {
              unsigned row = stride[0] * i + r - pad_top;
              unsigned col = stride[1] * j + c - pad_left;
              if (((col < xs[0]->d[1]) && (row < xs[0]->d[0]))) {
                if (!is_feasible) {
                  max_val = tb<3>(*xs[0])(row, col, ch, b);
                  max_r = row; max_c = col; is_feasible = true;
                } else if (tb<3>(*xs[0])(row, col, ch, b) > max_val) {
                  max_val = tb<3>(*xs[0])(row, col, ch, b);
                  max_r = row; max_c = col;
                }
              }
            }
          }
          (tb<3>(dEdxi))(max_r, max_c, ch, b) += (tb<3>(dEdf))(i, j, ch, b);
        }
      }
    }
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(MaxPooling2D)

} // namespace dynet
