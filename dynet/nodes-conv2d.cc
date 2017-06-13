#include "dynet/nodes-conv.h"

#include <algorithm>
#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <array>

#include "dynet/functors.h"
#include "dynet/nodes-macros.h"
#include "third_party/eigen_spatial_convolutions.h"
#include "third_party/eigen_backward_spatial_convolutions.h"
#include "third_party/eigen_pooling.h"

#if HAVE_CUDA
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
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
  if (xs[0].ndims() != 3 || xs[1].ndims() != 4 ||
      xs[1].d[2] != xs[0].d[2]) {
    ostringstream s; s << "Bad input dimensions in Conv2D: " << xs;
    throw std::invalid_argument(s.str());
  }
  if (is_valid && (xs[0].d[0] < xs[1].d[0] || xs[0].d[1] < xs[1].d[1])) {
    ostringstream s; s << "Bad input dimensions in Conv2D: in VALID convolution, the filter size must not be greater than the feature map size" << xs;
    throw std::invalid_argument(s.str());
  }
  if (xs.size() == 3) { //has bias term
    if (xs[2].d[0] != xs[1].d[3] || xs[2].ndims() != 1) {
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

size_t Conv2D::aux_storage_size() const {
  vector<unsigned> input_size(arity());
  for (unsigned i = 0; i < arity(); ++i) {
    input_size[i] = get_cg()->nodes[args[i]]->dim.size();
  }
  size_t nbytes = 0;
#if HAVE_CUDNN
  nbytes += CudnnConvOp::workspace_size_limit_bytes;
  nbytes += 3 * input_size[0] * sizeof(float);
#else
  nbytes += sizeof(float) * (input_size[0] + input_size[1] + 
      dim.size() + std::max(input_size[0], input_size[1]));
#endif
  return nbytes;
}
#endif

template<class MyDevice>
void Conv2D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2 || xs.size() == 3, "Failed dimension check in Conv2D::forward, at least 2 inputs");
  DYNET_ASSERT(fx.d.bd == xs[0]->d.bd, "Failed dimension check in Conv2D::forward, batchsize not match");
  DYNET_ASSERT(fx.d[2] == xs[1]->d[3], "Failed dimension check in Conv2D::forward, #channel not match");
  NodeMemPool aux_mem_pool = NodeMemPool(aux_storage_size(), aux_mem);
#ifdef __CUDACC__
#if HAVE_CUDNN
  if (cudnn_conv_op_ == NULL) {
    cudnn_conv_op_ = new CudnnConvOp(stride, is_valid);
    cudnn_conv_op_->set_pool(&aux_mem_pool);
  }
  cudnn_conv_op_->forward_impl(dev, xs, fx);
#else
  throw std::runtime_error("Conv2D::forward_dev_impl not supported without CUDNN");
#endif
#else
  std::cout << "Conv2D forward input" << endl;
  std::cout << "is_valid = " << is_valid << endl;
  std::cout << "dimensions are: " << endl;
  std::cout << "bd = " << xs[0]->d.bd << endl;
  std::cout << "d = " << xs[0]->d[2] << endl;
  std::cout << "i = " << xs[0]->d[0] << endl;
  std::cout << "j = " << xs[0]->d[1] << endl;
  std::cout << "Conv2D forward output" << endl;
  std::cout << "is_valid = " << is_valid << endl;
  std::cout << "dimensions are: " << endl;
  std::cout << "bd = " << fx.d.bd << endl;
  std::cout << "d = " << fx.d[2] << endl;
  std::cout << "i = " << fx.d[0] << endl;
  std::cout << "j = " << fx.d[1] << endl;
  //std::cout << "xs =" << *xs[0];

  Eigen::PaddingType padding_type = is_valid ? Eigen::PADDING_VALID : Eigen::PADDING_SAME;
  void* CHWN_x_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
  Tensor CHWN_x = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_x_mem), xs[0]->device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> shuffles; 
  shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
  CHWN_x.tb<3>().device(*dev.edevice) = xs[0]->tb<3>().shuffle(shuffles);
  void* NCHW_f_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
  Tensor NCHW_f = Tensor(Dim({xs[1]->d[3], xs[1]->d[2], xs[1]->d[0], xs[1]->d[1]}), static_cast<float*>(NCHW_f_mem), xs[1]->device, DeviceMempool::FXS);
  shuffles[0] = 3; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 1;
  NCHW_f.t<4>().device(*dev.edevice) = xs[1]->t<4>().shuffle(shuffles);
  void* CHWN_y_mem = aux_mem_pool.allocate(fx.d.size() * sizeof(float));
  Tensor CHWN_y = Tensor(Dim({fx.d[2], fx.d[0], fx.d[1]}, fx.d.bd), static_cast<float*>(CHWN_y_mem), fx.device, DeviceMempool::FXS);
  CHWN_y.tb<3>().device(*dev.edevice) = Eigen::SpatialConvolution(CHWN_x.tb<3>(), NCHW_f.t<4>(), stride[0], stride[1], padding_type);
  shuffles[0] = 1; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 3;
  fx.tb<3>().device(*dev.edevice) = CHWN_y.tb<3>().shuffle(shuffles);
  //NWHCToNCWH()(&NWHC_y, fx);
  if (xs.size() == 3) {
    Tensor bias = Tensor(Dim({fx.d[0], fx.d[1], fx.d.bd}, 1), static_cast<float*>(CHWN_x_mem), xs[2]->device, DeviceMempool::FXS);
    for (unsigned i = 0; i < fx.d[2]; ++i) {
      TensorTools::constant(bias, xs[2]->vec()(i));
      fx.tb<3>().chip<2>(i).device(*dev.edevice) += bias.t<3>(); 
    }
  }
  //std::cout << "fx =" << fx;
#endif
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
  NodeMemPool aux_mem_pool = NodeMemPool(aux_storage_size(), aux_mem);
#ifdef __CUDACC__
#if HAVE_CUDNN
  DYNET_ASSERT(cudnn_conv_op_ != NULL, "cudnn operator is not initialized");
  cudnn_conv_op_->set_pool(&aux_mem_pool);
  cudnn_conv_op_->backward_impl(dev, xs, fx, dEdf, i, dEdxi);
#else
  throw std::runtime_error("Conv2D::backward_dev_impl not supported without CUDNN");
#endif
#else
  void* CHWN_dy_mem = aux_mem_pool.allocate(dEdf.d.size() * sizeof(float));
  Tensor CHWN_dy = Tensor(Dim({dEdf.d[2], dEdf.d[0], dEdf.d[1]}, dEdf.d.bd), static_cast<float*>(CHWN_dy_mem), dEdf.device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> shuffles; 
  shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
  CHWN_dy.tb<3>().device(*dev.edevice) = dEdf.tb<3>().shuffle(shuffles);
  if (i == 0) { //backward w.r.t the input
    void* NCHW_f_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
    Tensor NCHW_f = Tensor(Dim({xs[1]->d[3], xs[1]->d[2], xs[1]->d[0], xs[1]->d[1]}), static_cast<float*>(NCHW_f_mem), xs[1]->device, DeviceMempool::FXS);
    shuffles[0] = 3; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 1;
    NCHW_f.t<4>().device(*dev.edevice) = xs[1]->t<4>().shuffle(shuffles);
    void* CHWN_dEdxi_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
    Tensor CHWN_dEdxi = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    CHWN_dEdxi.tb<3>().device(*dev.edevice) = Eigen::SpatialConvolutionBackwardInput(NCHW_f.t<4>(), CHWN_dy.tb<3>(), xs[0]->d[0], xs[0]->d[1], stride[0], stride[1]);
    void* HWCN_dEdxi_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
    Tensor HWCN_dEdxi = Tensor(xs[0]->d, static_cast<float*>(HWCN_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    shuffles[0] = 1; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 3;
    HWCN_dEdxi.tb<3>().device(*dev.edevice) = CHWN_dEdxi.tb<3>().shuffle(shuffles);
    dEdxi.tb<3>().device(*dev.edevice) += HWCN_dEdxi.tb<3>();
  } else if (i == 1) { //backward w.r.t the kernel
    void* CHWN_x_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
    Tensor CHWN_x = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_x_mem), xs[0]->device, DeviceMempool::FXS);
    shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
    CHWN_x.tb<3>().device(*dev.edevice) = xs[0]->tb<3>().shuffle(shuffles);
    void* NCHW_dEdxi_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
    Tensor NCHW_dEdxi = Tensor(Dim({xs[1]->d[3], xs[1]->d[2], xs[1]->d[0], xs[1]->d[1]}), static_cast<float*>(NCHW_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    NCHW_dEdxi.t<4>().device(*dev.edevice) = Eigen::SpatialConvolutionBackwardKernel(CHWN_x.tb<3>(), CHWN_dy.tb<3>(), xs[1]->d[0], xs[1]->d[1], stride[0], stride[1]);
    void* HWCN_dEdxi_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
    Tensor HWCN_dEdxi = Tensor(xs[1]->d, static_cast<float*>(HWCN_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    shuffles[0] = 2; shuffles[1] = 3; shuffles[2] = 1; shuffles[3] = 0;
    HWCN_dEdxi.t<4>().device(*dev.edevice) = NCHW_dEdxi.t<4>().shuffle(shuffles);
    dEdxi.t<4>().device(*dev.edevice) += HWCN_dEdxi.t<4>();
  } else { //backward w.r.t the bias
    Eigen::array<int, 3> red_axis = {0, 1, 3};
    dEdxi.t<1>().device(*dev.edevice) += dEdf.tb<3>().sum(red_axis);
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(Conv2D)


string MaxPool::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "maxpool(" << arg_names[0] << ")";
  return s.str();
}


Dim MaxPool::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1) {
    ostringstream s; s << "MaxPool requires exactly one input: " << xs;
    throw std::invalid_argument(s.str());
  }
  if (xs[0].ndims() != 3) {
    ostringstream s; s << "Bad input dimensions in MaxPool: " << xs;
    throw std::invalid_argument(s.str());
  }
  if (is_valid && (xs[0].d[0] < ksize[0] || xs[0].d[1] < ksize[1])) {
    ostringstream s; s << "Bad input dimensions in MaxPool: in VALID convolution, the kernel size must not be greater than the feature map size" << xs;
    throw std::invalid_argument(s.str());
  }
  unsigned bs = xs[0].batch_elems();
  std::vector<long> output_shape(3);
  output_shape[2] = static_cast<long>(xs[0].d[2]); //assumes #c is the same
						   //in input and output
  for (unsigned i = 0; i < 2; ++i) {
    float input_dim = static_cast<float>(xs[0].d[i]);
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



size_t MaxPool::aux_storage_size() const {
  vector<unsigned> input_size(arity());
  for (unsigned i = 0; i < arity(); ++i) {
    input_size[i] = get_cg()->nodes[args[i]]->dim.size();
  }
  size_t nbytes = 0;
  nbytes += sizeof(float) * (2*input_size[0] + 2*input_size[1] + dim.size());
  return nbytes;
}


template<class MyDevice>
void MaxPool::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in MaxPool::forward, exactly one input");
  DYNET_ASSERT(fx.d.bd == xs[0]->d.bd, "Failed dimension check in MaxPool::forward, batchsize not match");
  DYNET_ASSERT(fx.d[2] == xs[0]->d[2], "Failed dimension check in MaxPool::forward, #channel not match");
  NodeMemPool aux_mem_pool = NodeMemPool(aux_storage_size(), aux_mem);
  std::cout << "MaxPool forward input" << endl;
  std::cout << "is_valid = " << is_valid << endl;
  std::cout << "dimensions are: " << endl;
  std::cout << "bd = " << xs[0]->d.bd << endl;
  std::cout << "d = " << xs[0]->d[2] << endl;
  std::cout << "i = " << xs[0]->d[0] << endl;
  std::cout << "j = " << xs[0]->d[1] << endl;
  std::cout << "MaxPool forward output" << endl;
  std::cout << "is_valid = " << is_valid << endl;
  std::cout << "dimensions are: " << endl;
  std::cout << "bd = " << fx.d.bd << endl;
  std::cout << "d = " << fx.d[2] << endl;
  std::cout << "i = " << fx.d[0] << endl;
  std::cout << "j = " << fx.d[1] << endl;
#ifdef __CUDACC__
#if HAVE_CUDNN
  if (cudnn_conv_op_ == NULL) {
    cudnn_conv_op_ = new CudnnConvOp(stride, is_valid);
    cudnn_conv_op_->set_pool(&aux_mem_pool);
  }
  cudnn_conv_op_->forward_impl(dev, xs, fx);
#else
  throw std::runtime_error("MaxPool::forward_dev_impl not supported without CUDNN");
#endif
#else
  Eigen::PaddingType padding_type = is_valid ? Eigen::PADDING_VALID : Eigen::PADDING_SAME;
  void* CHWN_x_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
  Tensor CHWN_x = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_x_mem), xs[0]->device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> shuffles; 
  shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
  CHWN_x.tb<3>().device(*dev.edevice) = xs[0]->tb<3>().shuffle(shuffles);
  void* CHWN_y_mem = aux_mem_pool.allocate(fx.d.size() * sizeof(float));
  Tensor CHWN_y = Tensor(Dim({fx.d[2], fx.d[0], fx.d[1]}, fx.d.bd), static_cast<float*>(CHWN_y_mem), fx.device, DeviceMempool::FXS);
  CHWN_y.tb<3>().device(*dev.edevice) = Eigen::SpatialMaxPooling(CHWN_x.tb<3>(), ksize[0], ksize[1], stride[0], stride[1], padding_type);
  shuffles[0] = 1; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 3;
  fx.tb<3>().device(*dev.edevice) = CHWN_y.tb<3>().shuffle(shuffles);
#endif
}


template<class MyDevice>
void MaxPool::backward_dev_impl(const MyDevice & dev,
                         const vector<const Tensor*>& xs,
                         const Tensor& fx,
                         const Tensor& dEdf,
                         unsigned i,
                         Tensor& dEdxi) const {
  std::cout << "entered backward dev JAJAJAJAJA" << endl;
  // don't check those already checked in forward_impl
  DYNET_ASSERT(dEdf.d == fx.d, "Failed dimension check in MaxPool::backward");
  DYNET_ASSERT(dEdxi.d == xs[i]->d, "Failed dimension check in MaxPool::backward");
  DYNET_ASSERT(i == 0, "Failed dimension check in MaxPool::backward");
  NodeMemPool aux_mem_pool = NodeMemPool(aux_storage_size(), aux_mem);
#ifdef __CUDACC__
#if HAVE_CUDNN
  DYNET_ASSERT(cudnn_conv_op_ != NULL, "cudnn operator is not initialized");
  cudnn_conv_op_->set_pool(&aux_mem_pool);
  cudnn_conv_op_->backward_impl(dev, xs, fx, dEdf, i, dEdxi);
#else
  throw std::runtime_error("MaxPool::backward_dev_impl not supported without CUDNN");
#endif
#else


  if (is_valid) {
    //then fill it out with the correct result
    for (int b = 0; b < fx.d.bd; ++b) {
      for (int i = 0; i < fx.d[0]; ++i) {
        for (int j = 0; j < fx.d[1]; ++j) {
          for (int ch = 0; ch < fx.d[2]; ++ch) {
	    int largest_r = stride[0] * i;
            int largest_c = stride[1] * j;
            float largest = -10000.f;
            for (int r = 0; r < ksize[0]; ++r) {
              for (int c = 0; c < ksize[1]; ++c) {
                int row = stride[0] * i + r;
                int col = stride[1] * j + c;
                if ((col < xs[0]->d[1]) && (row < xs[0]->d[0])) {
                  if (xs[0]->tb<3>()(row, col, ch, b) > largest) {
                    largest = xs[0]->tb<3>()(row, col, ch, b);
                    largest_r = row;
                    largest_c = col;
                  }
                }
              }
            }
            (dEdxi.tb<3>())(largest_r, largest_c, ch, b) += (dEdf.tb<3>())(i, j, ch, b);
          }
        }
      }
    }
  } else {
    int in_height = xs[0]->d[0];
    int in_width = xs[1]->d[1];
    int filter_height = ksize[0];
    int filter_width = ksize[1];
    int out_height = ceil(float(in_height) / float(stride[0]));
    int out_width  = ceil(float(in_width) / float(stride[1]));
    int stride_rows = stride[0];
    int stride_cols = stride[1];
    int pad_along_height = ((out_height - 1) * stride_rows +
                    filter_height - in_height);
    int pad_along_width = ((out_width - 1) * stride_cols +
                   filter_width - in_width);
    int pad_top = pad_along_height / 2;
    int pad_left = pad_along_width / 2;
    for (int b = 0; b < fx.d.bd; ++b) {
      for (int i = 0; i < fx.d[0]; ++i) {
        for (int j = 0; j < fx.d[1]; ++j) {
          for (int ch = 0; ch < fx.d[2]; ++ch) {
	    int largest_r = stride[0] * i - pad_top;
            int largest_c = stride[1] * j - pad_left;
            float largest = -10000.f;
            for (int r = 0; r < filter_height; ++r) {
              for (int c = 0; c < filter_width; ++c) {
                int row = stride[0] * i + r - pad_top;
                int col = stride[1] * j + c - pad_left;
                if (((col < in_width) && (row < in_height)) && 
                   ((0 <= col) && (0 <= row))) {
                  if (xs[0]->tb<3>()(row, col, ch, b) > largest) {
                    largest = xs[0]->tb<3>()(row, col, ch, b);
                    largest_r = row;
                    largest_c = col;
                  }
                }
              }
            }
            std::cout << "i, j, largest_r, largest_c, largest" << 
			i << j << largest_r << largest_c << largest << endl;
            (dEdxi.tb<3>())(largest_r, largest_c, ch, b) += 
			(dEdf.tb<3>())(i, j, ch, b);
          }
        }
      }
    }
  }


   /* else {
    int in_rows = xs[0]->d[0];
    int in_cols = xs[0]->d[1];
    int window_rows = ksize[0];
    int window_cols = ksize[1];
    int row_stride = stride[0];
    int col_stride = stride[1];
    int out_rows = ceil(float(in_rows) / float(row_stride));
    int out_cols = ceil(float(in_cols) / float(col_stride));
    int pad_rows = ((out_rows - 1) * row_stride + window_rows - in_rows)/2;
    int pad_cols = ((out_cols - 1) * col_stride + window_cols - in_cols)/2;
    for (int b = 0; b < xs[0]->d.bd; ++b) {
      for (int h = 0; h < in_rows; ++h) {
        for (int w = 0; w < in_cols; ++h) {
          for (int ch = 0; ch < xs[0]->d[2]; ++ch) {
            int hpad = h + pad_rows;
            int wpad = w + pad_cols;
            int h_start = (hpad < window_rows) ? 0 : (hpad - window_rows) / row_stride + 1;
            int h_end = std::min(hpad / row_stride + 1, out_rows);
            int w_start = (wpad < window_cols) ? 0 : (wpad - window_cols) / col_stride + 1;
            int w_end = std::min(wpad / col_stride + 1, out_cols);
            //compute elementwise max
            int largest_r = h_start;
            int largest_c = w_start;
            int largest = -10000.f;
            for (int ph = h_start; ph < h_end; ++ph) {
              for (int pw = w_start; pw < w_end; ++pw) {
                if (largest < xs[0]->tb<3>()(ph, pw, ch, b)) {
                  largest = xs[0]->tb<3>()(ph, pw, ch, b);
                  largest_r = ph;
                  largest_c = pw;
                }
              }
            }
            (dEdxi.tb<3>())(largest_r, largest_c, ch, b) += (dEdf.tb<3>())(h, w, ch, b);
          }
        }
      }
    }
   

  } */


  /*
  else {
    int in_height = xs[0]->d[0];
    int in_width = xs[1]->d[1];
    int filter_height = ksize[0];
    int filter_width = ksize[1];
    int out_height = ceil(float(in_height) / float(stride[0]));
    int out_width  = ceil(float(in_width) / float(stride[1]));
    int pad_along_height = ((out_height - 1) * stride[0] +
                    filter_height - in_height);
    int pad_along_width = ((out_width - 1) * stride[1] +
                   filter_width - in_width);
    int pad_top = pad_along_height / 2;
    int pad_left = pad_along_width / 2;
    for (int b = 0; b < fx.d.bd; ++b) {
      for (int i = 0; i < fx.d[0]; ++i) {
        for (int j = 0; j < fx.d[1]; ++j) {
          for (int ch = 0; ch < fx.d[2]; ++ch) {
	    int largest_r = stride[0] * i - pad_top;
            int largest_c = stride[1] * j - pad_left;
            float largest = -10000.f;
            for (int r = 0; r < ksize[0]; ++r) {
              for (int c = 0; c < ksize[1]; ++c) {
                int row = stride[0] * i + r - pad_top;
                int col = stride[1] * j + c - pad_left;
                if (((col < in_width) && (row < in_height)) && 
                   ((0 <= col) && (0 <= row))) {
                  if (xs[0]->tb<3>()(row, col, ch, b) > largest) {
                    largest = xs[0]->tb<3>()(row, col, ch, b);
                    largest_r = row;
                    largest_c = col;
                  }
                }
              }
            }
            (dEdxi.tb<3>())(largest_r, largest_c, ch, b) += (dEdf.tb<3>())(i, j, ch, b);
          }
        }
      }
    }
  }
  */

  std::cout << "printing xs: " << endl;
  for (int b = 0; b < xs[0]->d.bd; ++b) {
    std::cout << "batch\n";
    for (int i = 0; i < xs[0]->d[0]; ++i) {
      for (int j = 0; j < xs[0]->d[1]; ++j) { 
        for (int ch = 0; ch < xs[0]->d[2]; ++ch) {
          std::cout << (xs[0]->tb<3>())(i, j, ch, b);
        }
      }
      std::cout << endl;
    }
  }

  std::cout << "printing fx: " << endl;
  for (int b = 0; b < fx.d.bd; ++b) {
    std::cout << "batch\n";
    for (int i = 0; i < fx.d[0]; ++i) {
      for (int j = 0; j < fx.d[1]; ++j) { 
        for (int ch = 0; ch < fx.d[2]; ++ch) {
          std::cout << (fx.tb<3>())(i, j, ch, b);
        }
      }
      std::cout << endl;
    }
  }

  std::cout << "printing dEdf: " << endl;
  for (int b = 0; b < dEdf.d.bd; ++b) {
    std::cout << "batch\n";
    for (int i = 0; i < dEdf.d[0]; ++i) {
      for (int j = 0; j < dEdf.d[1]; ++j) { 
        for (int ch = 0; ch < dEdf.d[2]; ++ch) {
          std::cout << (dEdf.tb<3>())(i, j, ch, b);
        }
      }
      std::cout << endl;
    }
  }

  std::cout << "printing dEdxi: " << endl;
  for (int b = 0; b < dEdxi.d.bd; ++b) {
    std::cout << "batch\n";
    for (int i = 0; i < dEdxi.d[0]; ++i) {
      for (int j = 0; j < dEdxi.d[1]; ++j) { 
        for (int ch = 0; ch < dEdxi.d[2]; ++ch) {
          std::cout << (dEdxi.tb<3>())(i, j, ch, b);
        }
      }
      std::cout << endl;
    }
  }


#endif
}
DYNET_NODE_INST_DEV_IMPL(MaxPool)

} // namespace dynet
