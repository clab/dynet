#include "dynet/nodes-conv.h"

#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <array>

#include "dynet/functors.h"
#include "dynet/nodes-macros.h"

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
  assert(xs.size() == 1 || xs.size() == 2);
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
    cerr << "Bad input dimensions in FoldRows: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in FoldRows");
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
    cerr << "Conv1DNarrow requires two inputs: " << xs << endl;
    throw std::invalid_argument("Conv1DNarrow requires 2 dimensions");
  }
  int ocols = xs[0].cols() - xs[1].cols() + 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() != 2 ||
      xs[0].rows() != xs[1].rows() ||
      ocols < 1) {
    cerr << "Bad input dimensions in Conv1DNarrow: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in Conv1DNarrow");
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
    cerr << "Conv1DWide requires two inputs: " << xs << endl;
    throw std::invalid_argument("Conv1DWide requires two inputs");
  }
  unsigned ocols = xs[0].cols() + xs[1].cols() - 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() != 2 ||
      xs[0].rows() != xs[1].rows()) {
    cerr << "Bad input dimensions in Conv1DWide: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in Conv1DWide");
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
    cerr << "Filter1DNarrow requires two inputs: " << xs << endl;
    throw std::invalid_argument("Filter1DNarrow requires 2 dimensions");
  }
  int ocols = xs[0].cols() - xs[1].cols() + 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() < 2 ||
      xs[0].rows() != xs[1].rows() ||
      ocols < 1) {
    cerr << "Bad input dimensions in Filter1DNarrow: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in Filter1DNarrow");
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
    cerr << "Bad bad k in KMaxPooling: " << k << endl;
    throw std::invalid_argument("bad k in KMaxPooling");
  }
  if (xs[0].ndims() != 2 || (xs[0].cols() < k)) {
    cerr << "Bad input dimensions in KMaxPooling: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in KMaxPooling");
  }
  return Dim({xs[0].rows(), k});
}

size_t KMaxPooling::aux_storage_size() const {
  // map of where the entries in f(x) go to entries in x
  return sizeof(Eigen::DenseIndex) * dim.size();
}

string SumColumns::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_cols(matrix=" << arg_names[0];
  if (arg_names.size() == 2) s << ", col_weighting=" << arg_names[1];
  s << ')';
  return s.str();
}

Dim SumColumns::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1 || xs.size() == 2);
  int bd = (xs.size() == 1 ? xs[0].bd : max(xs[0].bd, xs[1].bd));
  return Dim({xs[0].rows()}, bd);
}

#endif

template<class MyDevice>
void AverageColumns::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
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
  assert(i < 2);
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
void Filter1DNarrow::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const Eigen::array<Eigen::DenseIndex, 2> dims = {0, 1};
  if(xs[1]->d.ndims() == 2) {
    fx.t<2>().device(*dev.edevice) = xs[0]->t<2>().convolve(xs[1]->t<2>(), dims);
  } else {
    assert(xs[1]->d.ndims() > 2);
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
  assert(i < 2);
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
    assert(x.cols() < 1024);
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
          //assert(mi < dim.size());
          maxmap[mi++] = j;
          ++tt;
          if (tt == k) break;  // could happen in case of ties
        }
      }
    }
    assert(mi == dim.size());
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
  CUDA_CHECK(cudaMemcpyAsync((void*)maxmap, aux_mem, sizeof(Eigen::DenseIndex) * dim.size(), cudaMemcpyDeviceToHost));
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
void SumColumns::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#ifdef __CUDACC__
  // The reduction used on CPU is better, but not implemented in GPU
  unsigned cols = xs[0]->d.cols();
  fx.t<1>().device(*dev.edevice) = xs[0]->t<2>().chip<1>(0);
  for(unsigned i = 1; i < cols; ++i)
    fx.t<1>().device(*dev.edevice) += xs[0]->t<2>().chip<1>(i);
#else
  const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {1};
  fx.t<1>().device(*dev.edevice) = xs[0]->t<2>().sum(reduction_axis);
#endif
}

template<class MyDevice>
void SumColumns::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  const Eigen::array<Eigen::DenseIndex, 2> broadcasts = {1, xs[0]->d[1]};
  dEdxi.t<2>().device(*dev.edevice) += dEdf.t<2>().broadcast(broadcasts);
}
DYNET_NODE_INST_DEV_IMPL(SumColumns)

} // namespace dynet
