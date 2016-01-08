#include "cnn/conv.h"

#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>

#include "cnn/functors.h"
#if HAVE_CUDA
#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#endif

using namespace std;

namespace cnn {

string AddVectorToAllColumns::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "fold_rows(" << arg_names[0] << ", " << arg_names[1] << ')';
  return os.str();
}

Dim AddVectorToAllColumns::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 || xs[0].rows() != xs[1].rows() || xs[0].ndims() != 2 || xs[1].ndims() != 1) {
    cerr << "Bad input dimensions in AddVectorToAllColumns: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in AddVectorToAllColumns");
  }
  return xs[0];
}

void AddVectorToAllColumns::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
    throw std::runtime_error("AddVectorToAllColumns::forward not implemented for CUDA");
#else
  auto y = *fx;
  auto x = **xs[0];
  auto b = **xs[1];
  y = x.colwise() + b.col(0);
#endif
}

void AddVectorToAllColumns::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  assert(i < 2);
  if (i == 0) { // x
    (*dEdxi) += (*dEdf);
  } else { // bias
    (*dEdxi).col(0) += (*dEdf).rowwise().sum();
  }
}

string FoldRows::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "fold_rows(" << arg_names[0] << ", nrows=" << nrows << ')';
  return os.str();
}

Dim FoldRows::dim_forward(const vector<Dim>& xs) const {
  unsigned orows = xs[0].rows() / nrows;
  if ((orows * nrows != xs[0].rows()) || xs.size() != 1 || xs[0].ndims() != 2) {
    cerr << "Bad input dimensions in FoldRows: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in FoldRows");
  }
  return Dim({orows, xs[0].cols()});
}

void FoldRows::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("FoldRows::forward not implemented for CUDA");
#else
  auto x = **xs[0];
  auto y = *fx;
  unsigned orows = y.rows();
  for (unsigned i = 0; i < orows; ++i) {
    for (unsigned j = 0; j < nrows; ++j) {
      if (j)
        y.row(i) += x.row(i * nrows + j);
      else // j = 0
        y.row(i) = x.row(i * nrows);
    }
  }
#endif
}

void FoldRows::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("AddVectorToAllColumns::backward not implemented for CUDA");
#else
  unsigned orows = fx.d.rows();
  auto d = *dEdf;
  auto di = *dEdxi;
  for (unsigned i = 0; i < orows; ++i)
    for (unsigned j = 0; j < nrows; ++j)
      di.row(i * nrows + j) += d.row(i);
#endif
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
  unsigned ocols = xs[0].cols() - xs[1].cols() + 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() != 2 ||
      xs[0].rows() != xs[1].rows() ||
      ocols < 1) {
    cerr << "Bad input dimensions in Conv1DNarrow: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in Conv1DNarrow");
  }
  return Dim({xs[0].rows(), ocols});
}

void Conv1DNarrow::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Conv1DNarrow::forward not implemented for CUDA");
#else
  // TODO this is a bad implementation- rewrite to use unsupported Eigen tensor library
  auto x = **xs[0];  // input
  auto f = **xs[1];  // filter
  auto y = *fx;
  const unsigned rows = x.rows();
  const unsigned ycols = dim.cols();
  const unsigned fcols = f.cols();
  for (unsigned i = 0; i < rows; ++i) {
    for (unsigned j = 0; j < ycols; ++j) {
      float t = 0;
      for (unsigned k = 0; k < fcols; ++k)
        t += f(i, k) * x(i, j + k);
      y(i, j) = t;
    }
  }
#endif
}

void Conv1DNarrow::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Conv1DNarrow::backward not implemented for CUDA");
#else
  // TODO this is a bad implementation- rewrite to use unsupported Eigen tensor library
  assert(i < 2);
  const unsigned rows = xs[0]->d.rows();
  const unsigned ycols = dim.cols();
  const unsigned fcols = xs[1]->d.cols();
  auto d = *dEdf;
  auto di = *dEdxi;
  if (i == 0) { // derivative wrt input x
    auto f = **xs[1];
    for (unsigned i = 0; i < rows; ++i) {
      for (unsigned j = 0; j < ycols; ++j) {
        for (unsigned k = 0; k < fcols; ++k)
          di(i, j + k) += f(i, k) * d(i, j);
      }
    }
  } else { // derivative wrt filter f
    auto x = **xs[0];
    for (unsigned i = 0; i < rows; ++i) {
      for (unsigned j = 0; j < ycols; ++j) {
        for (unsigned k = 0; k < fcols; ++k)
          di(i, k) += x(i, j + k) * d(i, j);
      }
    }
  }
#endif
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

void Conv1DWide::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Conv1DWide::forward not implemented for CUDA");
#else
  TensorTools::Zero(fx);
  auto x = **xs[0];  // input
  auto f = **xs[1];  // filter
  auto y = *fx;
  const unsigned rows = x.rows();
  const unsigned xcols = x.cols();
  const unsigned fcols = f.cols();
  for (unsigned i = 0; i < rows; ++i) {
    for (unsigned j = 0; j < xcols; ++j) {
      const float xij = x(i, j);
      for (unsigned k = 0; k < fcols; ++k)
        y(i, j + k) += f(i, k) * xij;
    }
  }
#endif
}

void Conv1DWide::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Conv1DWide::backward not implemented for CUDA");
#else
  assert(i < 2);
  const unsigned rows = xs[0]->d.rows();
  const unsigned xcols = xs[0]->d.cols();
  const unsigned fcols = xs[1]->d.cols();
  auto d = *dEdf;
  auto di = *dEdxi;
  if (i == 0) { // derivative wrt input x
    auto f = **xs[1];
    for (unsigned i = 0; i < rows; ++i) {
      for (unsigned j = 0; j < xcols; ++j) {
        for (unsigned k = 0; k < fcols; ++k)
          di(i, j) += f(i, k) * d(i, j + k);
      }
    }
  } else { // derivative wrt filter f
    auto x = **xs[0];
    for (unsigned i = 0; i < rows; ++i) {
      for (unsigned j = 0; j < xcols; ++j) {
        const float xij = x(i, j);
        for (unsigned k = 0; k < fcols; ++k)
          di(i, k) += xij * d(i, j + k);
      }
    }
  }
#endif
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

void KMaxPooling::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("KMaxPooling::forward not implemented for CUDA");
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

void KMaxPooling::backward_impl(const vector<const Tensor*>& xs,
                           const Tensor& fx,
                           const Tensor& dEdf,
                           unsigned i,
                           Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("KMaxPooling::backward not implemented for CUDA");
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

} // namespace cnn
