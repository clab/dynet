#include "cnn/nodes-conv.h"

#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>

#include "cnn/functors.h"
#include "cnn/nodes-macros.h"

#if HAVE_CUDA
#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#endif

using namespace std;

namespace cnn {

#ifndef __CUDACC__

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
  return sizeof(int) * dim.size();
}

#endif

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
  array<ptrdiff_t, 1> broadcasts; broadcasts[0] = nrows;
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec().broadcast(broadcasts);
  // unsigned orows = fx.d.rows();
  // for (unsigned i = 0; i < orows; ++i)
  //   for (unsigned j = 0; j < nrows; ++j)
  //     dEdxi.tb<2>().chip<0>(i * nrows + j) += d.tb<2>().chip<0>(i);
}
CNN_NODE_INST_DEV_IMPL(FoldRows)

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
CNN_NODE_INST_DEV_IMPL(Conv1DNarrow)

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
CNN_NODE_INST_DEV_IMPL(Conv1DWide)

template<class MyDevice>
void KMaxPooling::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  throw std::runtime_error("KMaxPooling::forward_dev_impl not working on CUDA yet");
#else
  auto x=**xs[0];
  auto y=*fx;
  float tmp[1024];
  assert(x.cols() < 1024);
  unsigned mi = 0;
  const unsigned rows = x.rows();
  const unsigned xcols = x.cols();
  int* maxmap = static_cast<int*>(aux_mem);
  for (unsigned i=0; i < rows; ++i) {
    //cerr << "row(" << i << ")=" << x.row(i) << endl;
    for (unsigned j=0; j < xcols; ++j)
      tmp[j] = -x(i,j);
    nth_element(tmp, tmp + (k-1), tmp + xcols);
    const float c = -tmp[k-1];  // kth largest element in row i
    unsigned tt = 0;
    for (unsigned j = 0; j < xcols; ++j) {
      const float xij = x(i,j);
      if (xij >= c) {
        //cerr << xij << ' ';
        y(i,tt) = xij;
        //assert(mi < dim.size());
        maxmap[mi++] = j;
        ++tt;
        if (tt == k) break;  // could happen in case of ties
      }
    }
    //cerr << endl; abort();
  }
  assert(mi == dim.size());
#endif
}

template<class MyDevice>
void KMaxPooling::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  throw std::runtime_error("Conv1DWide::backward_dev_impl not working on CUDA yet");
#else
  const unsigned rows = dim.rows();
  const unsigned cols = dim.cols();
  const int* maxmap = static_cast<const int*>(aux_mem);
  for (unsigned i = 0; i < rows; ++i) {
    unsigned mi = 0;
    for (unsigned j = 0; j < cols; ++j) {
      assert(mi < dim.size());
      const int oj = maxmap[mi++];
      if (oj > (*dEdxi).cols() || oj < 0) {
        cerr << dim << (*fx) << endl << (*dEdxi) << endl;
        cerr << "MM:"; for (unsigned k=0;k < dim.size(); ++k) cerr << ' ' << maxmap[k];
        cerr << endl;
        cerr << "BAD: " << oj << endl; abort();
      }
      (*dEdxi)(i, oj) += (*dEdf)(i, j);
    }
  }
#endif
}
CNN_NODE_INST_DEV_IMPL(KMaxPooling)

} // namespace cnn
