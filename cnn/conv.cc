#include "cnn/conv.h"

#include <sstream>
#include <limits>
#include <cmath>

#include "cnn/functors.h"
#if HAVE_CUDA
#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#endif

using namespace std;

namespace cnn {

string Conv1DNarrow::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "conv1dnarrow(" << arg_names[0] << ", f=" << arg_names[1] << ')';
  return os.str();
}

Dim Conv1DNarrow::dim_forward(const vector<Dim>& xs) const {
  int ocols = xs[0].cols() - xs[1].cols() + 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() != 2 ||
      xs[0].rows() != xs[1].rows() ||
      ocols < 1) {
    cerr << "Bad input dimensions in Conv1DNarrow: " << xs << endl;
    abort();
  }
  return Dim({xs[0].rows(), ocols});
}

void Conv1DNarrow::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
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
}

void Conv1DNarrow::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
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
}

string KMaxPooling::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "kmaxpool(" << arg_names[0] << ", k=" << k << ')';
  return os.str();
}

Dim KMaxPooling::dim_forward(const vector<Dim>& xs) const {
  if (k < 1) {
    cerr << "Bad bad k in KMaxPooling: " << k << endl;
    abort();
  }
  if (xs[0].ndims() != 2 ||
      (xs[0].cols() < k)) {
    cerr << "Bad input dimensions in KMaxPooling: " << xs << endl;
    abort();
  }
  return Dim({xs[0].rows(), k});
}

size_t KMaxPooling::aux_storage_size() const {
  // map of where the entries in f(x) go to entries in x
  return sizeof(int) * dim.size();
}

void KMaxPooling::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x=**xs[0];
  auto y=*fx;
  float tmp[1024];
  assert(x.cols() < 1024);
  int mi = 0;
  const unsigned rows = x.rows();
  const unsigned xcols = x.cols();
  int* maxmap = static_cast<int*>(aux_mem);
  for (unsigned i=0; i < rows; ++i) {
    //cerr << "row(" << i << ")=" << x.row(i) << endl;
    for (unsigned j=0; j < xcols; ++j) 
      tmp[j] = -x(i,j);
    nth_element(tmp, tmp + (k-1), tmp + xcols);
    const float c = -tmp[k-1];  // kth largest element in row i
    int tt = 0;
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
}

void KMaxPooling::backward(const vector<const Tensor*>& xs,
                           const Tensor& fx,
                           const Tensor& dEdf,
                           unsigned i,
                           Tensor& dEdxi) const {
  const unsigned rows = dim.rows();
  const unsigned cols = dim.cols();
  const int* maxmap = static_cast<const int*>(aux_mem);
  for (unsigned i = 0; i < rows; ++i) {
    int mi = 0;
    for (unsigned j = 0; j < cols; ++j) {
      assert(mi < dim.size());
      const int oj = maxmap[mi++];
      (*dEdxi)(i, oj) += (*dEdf)(i, j);
    }
  }
}

} // namespace cnn
