#include "dynet/nodes-conv.h"

#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <array>

#include "dynet/functors.h"
#include "dynet/nodes-macros.h"
#include <dynet/eigen_spatial_convolutions.h>
#include <dynet/eigen_backward_spatial_convolutions.h>

#if HAVE_CUDA
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#endif

using namespace std;

namespace dynet {

#ifndef __CUDACC__

string AverageColumns::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "average_cols(matrix=" << arg_names[0] << ')';
  return s.str();
}

Dim AverageColumns::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1 || xs.size() == 2, "Failed input count check in AverageColumns");
  int bd = (xs.size() == 1 ? xs[0].bd : max(xs[0].bd, xs[1].bd));
  return Dim({xs[0].rows()}, bd);
}

string FoldRows::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "fold_rows(" << arg_names[0] << ", nrows=" << nrows << ')';
  return os.str();
}

Dim FoldRows::dim_forward(const vector<Dim>& xs) const {
  unsigned orows = xs[0].rows() / nrows;
  if ((orows * nrows != xs[0].rows()) || xs.size() != 1 || xs[0].ndims() > 2) {
    ostringstream s; s << "Bad input dimensions in FoldRows: " << xs;
    throw std::invalid_argument(s.str());
  }
  return Dim({orows, xs[0].cols()});
}

string Conv1DNarrow::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "conv1d_narrow(" << arg_names[0] << ", f=" << arg_names[1] << ')';
  return os.str();
}

Dim Conv1DNarrow::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2) {
    ostringstream s; s << "Conv1DNarrow requires two inputs: " << xs;
    throw std::invalid_argument(s.str());
  }
  int ocols = xs[0].cols() - xs[1].cols() + 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() != 2 ||
      xs[0].rows() != xs[1].rows() ||
      ocols < 1) {
    ostringstream s; s << "Bad input dimensions in Conv1DNarrow: " << xs;
    throw std::invalid_argument(s.str());
  }
  return Dim({xs[0].rows(), (unsigned)ocols});
}

string Conv1DWide::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "conv1d_wide(" << arg_names[0] << ", f=" << arg_names[1] << ')';
  return os.str();
}

Dim Conv1DWide::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2) {
    ostringstream s; s << "Conv1DWide requires two inputs: " << xs;
    throw std::invalid_argument(s.str());
  }
  unsigned ocols = xs[0].cols() + xs[1].cols() - 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() != 2 ||
      xs[0].rows() != xs[1].rows()) {
    ostringstream s; s << "Bad input dimensions in Conv1DWide: " << xs;
    throw std::invalid_argument(s.str());
  }
  return Dim({xs[0].rows(), ocols});
}

string Filter1DNarrow::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "conv1d_narrow(" << arg_names[0] << ", f=" << arg_names[1] << ')';
  return os.str();
}

Dim Filter1DNarrow::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2) {
    ostringstream s; s << "Filter1DNarrow requires two inputs: " << xs;
    throw std::invalid_argument(s.str());
  }
  int ocols = xs[0].cols() - xs[1].cols() + 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() < 2 ||
      xs[0].rows() != xs[1].rows() ||
      ocols < 1) {
    ostringstream s; s << "Bad input dimensions in Filter1DNarrow: " << xs;
    throw std::invalid_argument(s.str());
  }
  const unsigned fids = (xs[1].ndims() > 2 ? xs[1][2] : 1);
  return Dim({fids, (unsigned)ocols});
}

string KMaxPooling::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "kmaxpool(" << arg_names[0] << ", k=" << k << ')';
  return os.str();
}

Dim KMaxPooling::dim_forward(const vector<Dim>& xs) const {
  if (k < 1) {
    ostringstream s; s << "Bad bad k in KMaxPooling: " << k;
    throw std::invalid_argument(s.str());
  }
  if (xs[0].ndims() != 2 || (xs[0].cols() < k)) {
    ostringstream s; s << "Bad input dimensions in KMaxPooling: " << xs;
    throw std::invalid_argument(s.str());
  }
  return Dim({xs[0].rows(), k});
}

size_t KMaxPooling::aux_storage_size() const {
  // map of where the entries in f(x) go to entries in x
  return sizeof(Eigen::DenseIndex) * dim.size();
}

string SumDimension::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_dim(matrix=" << arg_names[0] << ',' << dimension << '}';
  return s.str();
}

Dim SumDimension::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");
  Dim ret(xs[0]);
  ret.delete_dim(dimension);
  return ret;
}

string Conv2D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "conv2d(" << arg_names[0] << ", f=" << arg_names[1] << ')';
  return s.str();
}

Dim Conv2D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2) {
    ostringstream s; s << "Conv2D requires two inputs: " << xs;
    throw std::invalid_argument(s.str());
  }
  // check inputs and filters
  if (xs[0].ndims() != 3 || xs[1].ndims() != 4 ||
      xs[1].d[1] != xs[0].d[0] ) {
    ostringstream s; s << "Bad input dimensions in Conv2D: " << xs;
    throw std::invalid_argument(s.str());
  }

  unsigned bs = xs[0].batch_elems();
  std::vector<long> output_shape(3);
  output_shape[0] = (long)xs[1].d[0];
  // the first tensor is batched data (N * (C * H * W)), the second tensor is the filters (Co* Ci * H * W)
  for (unsigned i = 0; i < 2; ++i ){
    // skip i = 0, which is the channel dimension
    unsigned input_dim = xs[0].d[i+1];
    unsigned kernel_dim = xs[1].d[i+2];
    if (is_valid) {
      output_shape[i+1] = (long)((float)(input_dim - kernel_dim + 1) / (float)stride[i]);
    } else {
      output_shape[i+1] = (long)((float)input_dim / (float)stride[i]);
    }
  }
  return Dim(output_shape, bs);
}
#endif

template<class MyDevice>
void AverageColumns::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in AverageColumns");
  unsigned cols = xs[0]->d.cols();
#ifdef __CUDACC__
  // The reduction used on CPU is better, but not implemented in GPU
  fx.t<1>().device(*dev.edevice) = xs[0]->t<2>().chip<1>(0);
  for(unsigned i = 1; i < cols; ++i)
    fx.t<1>().device(*dev.edevice) += xs[0]->t<2>().chip<1>(i);
  fx.t<1>().device(*dev.edevice) = fx.t<1>() / (float)cols;
#else
  const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {1};
  fx.t<1>().device(*dev.edevice) = xs[0]->t<2>().sum(reduction_axis) / (float)cols;
#endif
}

template<class MyDevice>
void AverageColumns::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  const Eigen::array<Eigen::DenseIndex, 2> broadcasts = {1, xs[0]->d[1]};
  dEdxi.t<2>().device(*dev.edevice) += (dEdf.t<2>() / (float)xs[0]->d[1]).broadcast(broadcasts);
}
DYNET_NODE_INST_DEV_IMPL(AverageColumns)

template<class MyDevice>
void Conv1DNarrow::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned ycols = dim.cols();
  const unsigned fcols = xs[1]->d.cols();
  for (unsigned j = 0; j < ycols; ++j) {
    fx.t<2>().chip<1>(j).device(*dev.edevice) = xs[0]->t<2>().chip<1>(j) * xs[1]->t<2>().chip<1>(0);
    for (unsigned k = 1; k < fcols; ++k)
      fx.t<2>().chip<1>(j).device(*dev.edevice) += xs[0]->t<2>().chip<1>(j+k) * xs[1]->t<2>().chip<1>(k);
  }
  // TODO: This following version without chip is better, but for some reason dimensions don't match.
  // Eigen::array<ptrdiff_t, 1> dims; dims[0] = 1;
  // fx.t<2>().device(*dev.edevice) = xs[0]->t<2>().convolve(xs[1]->t<2>(), dims);
}

template<class MyDevice>
void Conv1DNarrow::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed input count check in Conv1DNarrow");
  const unsigned ycols = dim.cols();
  const unsigned fcols = xs[1]->d.cols();
  // TODO: Can this be done with a kernel and without using chip?
  if (i == 0) { // derivative wrt input x
    for (unsigned j = 0; j < ycols; ++j)
      for (unsigned k = 0; k < fcols; ++k)
        dEdxi.t<2>().chip<1>(j+k).device(*dev.edevice) += xs[1]->t<2>().chip<1>(k) * dEdf.t<2>().chip<1>(j);
  } else { // derivative wrt filter f
    for (unsigned j = 0; j < ycols; ++j)
      for (unsigned k = 0; k < fcols; ++k)
        dEdxi.t<2>().chip<1>(k).device(*dev.edevice) += xs[0]->t<2>().chip<1>(j+k) * dEdf.t<2>().chip<1>(j);
  }
}
DYNET_NODE_INST_DEV_IMPL(Conv1DNarrow)

template<class MyDevice>
void Conv1DWide::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  TensorTools::Zero(fx);
  const unsigned xcols = xs[0]->d.cols();
  const unsigned fcols = xs[1]->d.cols();
  for (unsigned j = 0; j < xcols; ++j)
    for (unsigned k = 0; k < fcols; ++k)
      fx.t<2>().chip<1>(j+k).device(*dev.edevice) += xs[1]->t<2>().chip<1>(k) * xs[0]->t<2>().chip<1>(j);
}


template<class MyDevice>
void Conv1DWide::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  const unsigned xcols = xs[0]->d.cols();
  const unsigned fcols = xs[1]->d.cols();
  if (i == 0) { // derivative wrt input x
    for (unsigned j = 0; j < xcols; ++j)
      for (unsigned k = 0; k < fcols; ++k)
        dEdxi.t<2>().chip<1>(j).device(*dev.edevice) += xs[1]->t<2>().chip<1>(k) * dEdf.t<2>().chip<1>(j + k);
  } else { // derivative wrt filter f
    for (unsigned j = 0; j < xcols; ++j)
      for (unsigned k = 0; k < fcols; ++k)
        dEdxi.t<2>().chip<1>(k).device(*dev.edevice) += xs[0]->t<2>().chip<1>(j) * dEdf.t<2>().chip<1>(j + k);
  }
}
DYNET_NODE_INST_DEV_IMPL(Conv1DWide)

template<class MyDevice>
void Conv2D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in Conv2D::forward");
  DYNET_ASSERT(fx.d.bd == xs[0]->d.bd, "Failed dimension check in Conv2D::forward");
  DYNET_ASSERT(fx.d[0] == xs[1]->d[0], "Failed dimension check in Conv2D::forward");
  Eigen::PaddingType padding_type = is_valid ? Eigen::PADDING_VALID : Eigen::PADDING_SAME;
#ifdef __CUDACC__
  //TODO (Hao Zhang): implement a CUDNN version
  throw std::runtime_error("Conv2D::forward_dev_impl not working on CUDA yet");
#else
  // this is the fastest (available) implementation as reported by TensorFlow community on CPU
  fx.tb<3>().device(*dev.edevice) = Eigen::SpatialConvolution(xs[0]->tb<3>(), xs[1]->t<4>(), stride[0], stride[1], padding_type);
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
  DYNET_ASSERT(i < 2, "Failed dimension check in Conv2D::backward");
#ifdef __CUDACC__
  //TODO (Hao Zhang): implement a version using CUDNN
  throw std::runtime_error("Conv2D::backward_dev_impl not working on CUDA yet");
#else
  if (i == 0) { //backward w.r.t the input
    dEdxi.tb<3>().device(*dev.edevice) += Eigen::SpatialConvolutionBackwardInput(xs[1]->t<4>(), dEdf.tb<3>(), xs[0]->d[1], xs[0]->d[2], stride[0], stride[1]);
  } else { //backward w.r.t the kernel
    dEdxi.t<4>().device(*dev.edevice) += Eigen::SpatialConvolutionBackwardKernel(xs[0]->tb<3>(), dEdf.tb<3>(), xs[1]->d[2], xs[1]->d[3], stride[0], stride[1]);
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(Conv2D)

template<class MyDevice>
void Filter1DNarrow::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const Eigen::array<Eigen::DenseIndex, 2> dims = {0, 1};
  if(xs[1]->d.ndims() == 2) {
    fx.t<2>().device(*dev.edevice) = xs[0]->t<2>().convolve(xs[1]->t<2>(), dims);
  } else {
    DYNET_ASSERT(xs[1]->d.ndims() > 2, "Input to Filter1DNarrow must have 2 or more dimensions");
    const unsigned fids = xs[1]->d[2];
    const unsigned ycols = dim.cols();
    Eigen::DSizes<ptrdiff_t, 2> indices(0,0);
    Eigen::DSizes<ptrdiff_t, 2> sizes(1,ycols);
    for(unsigned fid = 0; fid < fids; ++fid) {
      indices[0] = fid;
#if defined(__CUDACC__) && defined(EIGEN_NO_MALLOC)
      throw std::runtime_error("CUDA memory allocation in Filter1DNarrow");
#endif
      fx.t<2>().slice(indices, sizes).device(*dev.edevice) = xs[0]->t<2>().convolve(xs[1]->t<3>().chip<2>(fid), dims);
    }
  }
}

template<class MyDevice>
void Filter1DNarrow::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed input count check in Filter1DNarrow");
  const unsigned rows = xs[1]->d.rows();
  const unsigned ycols = dim.cols();
  const unsigned fcols = xs[1]->d.cols();
  const unsigned fids = (xs[1]->d.ndims() > 2 ? xs[1]->d[2] : 1);
  Eigen::DSizes<ptrdiff_t, 2> sizes(rows,fcols);
  Eigen::DSizes<ptrdiff_t, 2> indices(0,0);
  // TODO: This implementation is by no means optimized. Is there a better way to do it?
  vector<float> dEdf_vec = as_vector(dEdf);
  if(i == 0) {
    for(unsigned i = 0; i < ycols; i++) {
      indices[1] = i;
      if(fids == 1) {
        dEdxi.t<2>().slice(indices, sizes).device(*dev.edevice) += xs[1]->t<2>() * dEdf_vec[i];
      } else {
        for(unsigned fid = 0; fid < fids; fid++)
          dEdxi.t<2>().slice(indices, sizes).device(*dev.edevice) += xs[1]->t<3>().chip<2>(fid) * dEdf_vec[fid + i * fids];
      }
    }
  } else {
    for(unsigned i = 0; i < ycols; i++) {
      indices[1] = i;
      if(fids == 1) {
        dEdxi.t<2>().device(*dev.edevice) += xs[0]->t<2>().slice(indices, sizes) * dEdf_vec[i];
      } else {
        for(unsigned fid = 0; fid < fids; fid++)
          dEdxi.t<3>().chip<2>(fid).device(*dev.edevice) += xs[0]->t<2>().slice(indices, sizes) * dEdf_vec[fid + i * fids];
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(Filter1DNarrow)


template<class MyDevice>
void FoldRows::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned orows = fx.d.rows();
  for (unsigned i = 0; i < orows; ++i) {
    fx.tb<2>().chip<0>(i).device(*dev.edevice) = xs[0]->tb<2>().chip<0>(i * nrows);
    for (unsigned j = 1; j < nrows; ++j) 
      fx.tb<2>().chip<0>(i).device(*dev.edevice) += xs[0]->tb<2>().chip<0>(i * nrows + j); 
  }
  // TODO: This broadcasting should work?
  // array<ptrdiff_t, 1> broadcasts; broadcasts[0] = nrows;
  // fx.tvec().broadcast(broadcasts).device(*dev.edevice) += xs[0]->tvec();
}

template<class MyDevice>
void FoldRows::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  const Eigen::array<Eigen::DenseIndex, 1> broadcasts = {nrows};
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec().broadcast(broadcasts);
  // unsigned orows = fx.d.rows();
  // for (unsigned i = 0; i < orows; ++i)
  //   for (unsigned j = 0; j < nrows; ++j)
  //     dEdxi.tb<2>().chip<0>(i * nrows + j) += d.tb<2>().chip<0>(i);
}
DYNET_NODE_INST_DEV_IMPL(FoldRows)

template<class MyDevice>
void KMaxPooling::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::DenseIndex* maxmap = static_cast<Eigen::DenseIndex*>(aux_mem);
  if(k == 1) {
    Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex,1>> locs(maxmap, dim.size());
    locs.device(*dev.edevice) = xs[0]->t<2>().argmax(1);
#ifdef __CUDACC__
    // TODO: The code that works on CPU does not compile on CUDA
    throw std::runtime_error("KMaxPooling::forward_dev_impl not working on CUDA yet");
#else
    const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {1};
    fx.t<1>().device(*dev.edevice) = xs[0]->t<2>().maximum(reduction_axis);
#endif
  } else {
#ifdef __CUDACC__
    // TODO: Can this be done by CUDNN?
    throw std::runtime_error("KMaxPooling::forward_dev_impl for k>1 not working on CUDA yet");
#else
    auto x=**xs[0];
    auto y=*fx;
    float tmp[1024];
    DYNET_ASSERT(x.cols() < 1024, "KMaxPooling only works for expressions of size < 1024 at the moment. Got " << x.cols());
    unsigned mi = 0;
    const unsigned rows = x.rows();
    const unsigned xcols = x.cols();
    for (unsigned i=0; i < rows; ++i) {
      for (unsigned j=0; j < xcols; ++j)
        tmp[j] = -x(i,j);
      nth_element(tmp, tmp + (k-1), tmp + xcols);
      const float c = -tmp[k-1];  // kth largest element in row i
      unsigned tt = 0;
      for (unsigned j = 0; j < xcols; ++j) {
        const float xij = x(i,j);
        if (xij >= c) {
          y(i,tt) = xij;
          maxmap[mi++] = j;
          ++tt;
          if (tt == k) break;  // could happen in case of ties
        }
      }
    }
    DYNET_ASSERT(mi == dim.size(), "Programming error in KMaxPooling (mi != dim.size())");
#endif
  }
}

template<class MyDevice>
void KMaxPooling::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  // TODO: This is not at all efficient on GPU. Switch to CUDNN?
#ifdef __CUDACC__
  vector<Eigen::DenseIndex> indices(dim.size());
  const Eigen::DenseIndex* maxmap = &indices[0];
  CUDA_CHECK(cudaMemcpy((void*)maxmap, aux_mem, sizeof(Eigen::DenseIndex) * dim.size(), cudaMemcpyDeviceToHost));
#else
  const Eigen::DenseIndex* maxmap = static_cast<const Eigen::DenseIndex*>(aux_mem);
#endif
  const unsigned rows = dim.rows();
  for(unsigned i = 0; i < rows; ++i)
    for(unsigned j = 0; j < k; ++j, ++maxmap)
      dEdxi.t<2>().chip<1>(*maxmap).chip<0>(i).device(*dev.edevice) += dEdf.t<2>().chip<1>(j).chip<0>(i);
}
DYNET_NODE_INST_DEV_IMPL(KMaxPooling)

template<class MyDevice>
void SumDimension::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SumDimension");
  Eigen::array<int, 1> reduction_axis = {(int)dimension};
  fx.t<1>().device(*dev.edevice) = xs[0]->t<2>().sum(reduction_axis);
}

template<class MyDevice>
void SumDimension::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  // TODO: limit to 3-dimensional tensor is arbitrary
  Eigen::array<int, 4> bcast = {1,1,1,1}; bcast[dimension] = dEdxi.d[dimension];
  Eigen::array<int, 4> morph = {(int)dEdxi.d[0],(int)dEdxi.d[1],(int)dEdxi.d[2],(int)dEdxi.d.bd}; morph[dimension] = 1;
  dEdxi.tb<3>().device(*dev.edevice) += dEdf.tb<3>().reshape(morph).broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(SumDimension)

} // namespace dynet
