#include "dynet/tensor-eigen.h"
#include "dynet/nodes-conv.h"

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
#endif

using namespace std;

namespace dynet {

// ************* Filter1DNarrow *************

#ifndef __CUDACC__

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

#endif

template<class MyDevice>
void Filter1DNarrow::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const Eigen::array<Eigen::DenseIndex, 2> dims = {0, 1};
  if(xs[1]->d.ndims() == 2) {
    t<2>(fx).device(*dev.edevice) = t<2>(*xs[0]).convolve(t<2>(*xs[1]), dims);
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
      t<2>(fx).slice(indices, sizes).device(*dev.edevice) = t<2>(*xs[0]).convolve(t<3>(*xs[1]).chip<2>(fid), dims);
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
        t<2>(dEdxi).slice(indices, sizes).device(*dev.edevice) += t<2>(*xs[1]) * dEdf_vec[i];
      } else {
        for(unsigned fid = 0; fid < fids; fid++)
          t<2>(dEdxi).slice(indices, sizes).device(*dev.edevice) += t<3>(*xs[1]).chip<2>(fid) * dEdf_vec[fid + i * fids];
      }
    }
  } else {
    for(unsigned i = 0; i < ycols; i++) {
      indices[1] = i;
      if(fids == 1) {
        t<2>(dEdxi).device(*dev.edevice) += t<2>(*xs[0]).slice(indices, sizes) * dEdf_vec[i];
      } else {
        for(unsigned fid = 0; fid < fids; fid++)
          t<3>(dEdxi).chip<2>(fid).device(*dev.edevice) += t<2>(*xs[0]).slice(indices, sizes) * dEdf_vec[fid + i * fids];
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(Filter1DNarrow)

// ************* FoldRows *************

#ifndef __CUDACC__

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

#endif

template<class MyDevice>
void FoldRows::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned orows = fx.d.rows();
  for (unsigned i = 0; i < orows; ++i) {
    tb<2>(fx).chip<0>(i).device(*dev.edevice) = tb<2>(*xs[0]).chip<0>(i * nrows);
    for (unsigned j = 1; j < nrows; ++j)
      tb<2>(fx).chip<0>(i).device(*dev.edevice) += tb<2>(*xs[0]).chip<0>(i * nrows + j);
  }
  // TODO: This broadcasting should work?
  // array<ptrdiff_t, 1> broadcasts; broadcasts[0] = nrows;
  // tvec(fx).broadcast(broadcasts).device(*dev.edevice) += tvec(*xs[0]);
}

template<class MyDevice>
void FoldRows::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  //const Eigen::array<Eigen::DenseIndex, 1> broadcasts = {nrows};
  //tvec(dEdxi).device(*dev.edevice) += tvec(dEdf).broadcast(broadcasts);
  unsigned orows = fx.d.rows();
  for (unsigned i = 0; i < orows; ++i)
    for (unsigned j = 0; j < nrows; ++j)
      tb<2>(dEdxi).chip<0>(i * nrows + j).device(*dev.edevice) += tb<2>(dEdf).chip<0>(i);
}
DYNET_NODE_INST_DEV_IMPL(FoldRows)

// ************* KMaxPooling *************

#ifndef __CUDACC__

string KMaxPooling::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "kmaxpool(" << arg_names[0] << ", k=" << k << ", d=" << pooled_dim << ')';
  return os.str();
}

Dim KMaxPooling::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(pooled_dim < xs[0].nd,
                          "Tried to MaxDimension on dimension " << pooled_dim << " bigger than input " << xs[0]);
  DYNET_ARG_CHECK(xs[0].nd < 4,
                          "MaxDimension not currently supported for tensors of 4 or more dimensions.");
  DYNET_ARG_CHECK(k >= 1, "Bad bad k in KMaxPooling: " << k);
  DYNET_ARG_CHECK(k <= xs[0][pooled_dim],
                          "Bad k in KMaxPooling: k = " << k << " bigger than the size of pooled dimension "
                          << pooled_dim << " with size = " << xs[0][pooled_dim]);
  Dim ret(xs[0]);
  ret.set(pooled_dim, k);
  return ret;
}

size_t KMaxPooling::aux_storage_size() const {
  // map of where the entries in f(x) go to entries in x
  return sizeof(Eigen::DenseIndex) * dim.size();
}

#endif

template<class MyDevice>
void KMaxPooling::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
    // TODO: The code that works on CPU does not compile on CUDA
    throw std::runtime_error("KMaxPooling::forward_dev_impl not working on CUDA yet");
#endif
  Eigen::DenseIndex* maxmap = static_cast<Eigen::DenseIndex*>(aux_mem);
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>> locs(maxmap, dim[0], dim[1], dim[2], dim.batch_elems());
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[first_dim];
  const unsigned second_dim_size = dim[second_dim];
  Eigen::Tensor<float, 1> tmp(xs[0]->d[pooled_dim]);
  for (unsigned b = 0; b < batch_size; ++b){
    for (unsigned j = 0; j < second_dim_size; ++j){
      for (unsigned i = 0; i < first_dim_size; ++i){
        // get nth element
        tmp.device(*dev.edevice) = tb<3>(*xs[0]).chip<3>(b).chip(j, second_dim).chip(i, first_dim);
        nth_element(tmp.data(), tmp.data()+(k-1), tmp.data()+tmp.size(), std::greater<float>());
        const float c = tmp.data()[k-1];
        // calculate fx and indices
        tmp.device(*dev.edevice) = tb<3>(*xs[0]).chip<3>(b).chip(j, second_dim).chip(i, first_dim);
        unsigned tt = 0;
        for (unsigned l = 0; l < tmp.size(); ++l) {
          const float tensor_val = tmp.data()[l];
          if (tensor_val >= c) {
            if (pooled_dim > second_dim){
              tb<3>(fx).chip<3>(b).chip(tt, pooled_dim).chip(j, second_dim).chip(i, first_dim).device(*dev.edevice) = tmp.chip<0>(l);
              locs(i, j, tt, b) = l;
            }
            else if (pooled_dim > first_dim){
              tb<3>(fx).chip<3>(b).chip(j, second_dim).chip(tt, pooled_dim).chip(i, first_dim).device(*dev.edevice) = tmp.chip<0>(l);
              locs(i, tt, j, b) = l;
            }
            else {
              tb<3>(fx).chip<3>(b).chip(j, second_dim).chip(i, first_dim).chip(tt, pooled_dim).device(*dev.edevice) = tmp.chip<0>(l);
              locs(tt, i, j, b) = l;
            }
            ++tt;
            if (tt == k) break;  // could happen in case of ties
          }
        }
      }
    }
  }
}

template<class MyDevice>
void KMaxPooling::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i == 0, "Failed dimension check in KMaxPooling::backward");
#ifdef __CUDACC__
  vector<Eigen::DenseIndex> indices(dim.size());
  Eigen::DenseIndex* maxmap = &indices[0];
  CUDA_CHECK(cudaMemcpy((void*)maxmap, aux_mem, sizeof(Eigen::DenseIndex) * dim.size(), cudaMemcpyDeviceToHost));
#else
  Eigen::DenseIndex* maxmap = static_cast<Eigen::DenseIndex*>(aux_mem);
#endif
  Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>> locs(maxmap, dim[0], dim[1], dim[2], dim.batch_elems());
  const unsigned batch_size = dim.batch_elems();
  const unsigned first_dim_size = dim[first_dim];
  const unsigned second_dim_size = dim[second_dim];
  const unsigned pooled_dim_size = dim[pooled_dim];
  for(unsigned b = 0; b < batch_size; ++b){
    for(unsigned j = 0; j < second_dim_size; ++j){
      for(unsigned i = 0; i < first_dim_size; ++i){
        for(unsigned l = 0; l < pooled_dim_size; ++l){
          if (pooled_dim > second_dim)
            tb<3>(dEdxi).chip<3>(b).chip(locs(i, j, l, b), pooled_dim).chip(j, second_dim).chip(i, first_dim).device(*dev.edevice)
              += tb<3>(dEdf).chip<3>(b).chip<2>(l).chip<1>(j).chip<0>(i);
          else if (pooled_dim > first_dim)
            tb<3>(dEdxi).chip<3>(b).chip(j, second_dim).chip(locs(i, l, j, b), pooled_dim).chip(i, first_dim).device(*dev.edevice)
              += tb<3>(dEdf).chip<3>(b).chip<2>(j).chip<1>(l).chip<0>(i);
          else
            tb<3>(dEdxi).chip<3>(b).chip(j, second_dim).chip(i, first_dim).chip(locs(l, i, j, b), pooled_dim).device(*dev.edevice)
              += tb<3>(dEdf).chip<3>(b).chip<2>(j).chip<1>(i).chip<0>(l);
        }
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(KMaxPooling)

// ************* KMHNgram *************

#ifndef __CUDACC__

string KMHNGram::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "kmh-ngram(" << arg_names[0] << ')';
  return s.str();
}

Dim KMHNGram::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs[0].ndims() == 2, "Bad input dimensions in KMHNGram: " << xs);
  const unsigned new_cols = xs[0].cols() - n + 1;
  DYNET_ARG_CHECK(new_cols >= 1, "Bad input dimensions in KMHNGram: " << xs);
  return Dim({xs[0][0], new_cols});
}

#endif

template<class MyDevice>
void KMHNGram::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("KMHNGram forward");
#else
  auto x = mat(*xs[0]);
  const int new_cols = x.cols() - n + 1;
  DYNET_ASSERT(new_cols > 0, "Failed dimension check in KMHNGram");
  auto res = mat(fx);
  res.setZero();
  for (int j = 0; j < new_cols; ++j) {
    auto c_j = res.col(j);
    for (unsigned k = 0; k < n; ++k)
      c_j += x.col(j + k);
  }
#endif
}

template<class MyDevice>
void KMHNGram::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("KMHNGram backward");
#else
  const int c = dEdf.d.cols();
  for (int j = 0; j < c; ++j)
    for (unsigned k = 0; k < n; ++k)
      (mat(dEdxi)).col(j+k) += (mat(dEdf)).col(j);
#endif
}
DYNET_NODE_INST_DEV_IMPL(KMHNGram)

// ************* CircularCorrelation *************

#ifndef __CUDACC__

string CircularCorrelation::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "circ_corr(" << arg_names[0] << ", " << arg_names[1] << ')';
  return s.str();
}

Dim CircularCorrelation::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs[0].ndims() == 1 && xs[0] == xs[1],
                  "Bad input dimensions in CircularCorrelation: " << xs);
  return xs[0];
}

size_t CircularCorrelation::aux_storage_size() const {
  return dim.size() * sizeof(std::complex<float>) * 2;
}

#endif

namespace {
// These operations are implemented using FFTs (described below), but the slow
// quadratic implementation is left here for a reference.
inline int mod(int diff, int d) {
  if (diff < 0) return diff + d;
  return diff;
}

// adds the result of a * b to out
template <typename T>
void circular_convolution_naive(const T& a, const T& b, T& out) {
  const int d = a.dimension(0);
  for (int k = 0; k < d; ++k) {
    out(k) = 0;
    for (int i = 0; i < d; ++i) {
      out(k) += a(i) * b(mod(k - i, d));
    }
  }
}

// adds the result of a \star b to out
template <typename T>
void circular_correlation_naive(const T& a, const T& b, T& out) {
  const int d = a.dimension(0);
  for (int k = 0; k < d; ++k) {
    out(k) = 0;
    for (int i = 0; i < d; ++i) {
      out(k) += a(i) * b((k + i) % d);
    }
  }
}
}  // namespace

// The circular convolution operation a * b is computed as
//     IFFT(FFT(a) . FFT(b)),
// where . is componentwise multiplication of complex numbers. Since a and b
// are real vectors (DyNet doesn't do complex numbers), the result of the
// convolution is also real valued, so when computing the IFFT, it is only
// necessary to compute the real part.
//
// The circular correlation operation a \star b is computed as
//     IFFT(conj(FFT(a)) . FFT(b)),
// where . is componentwise complex multiplication, and conj is the complex
// conjugate. Again, the IFFT need only compute the real part since the
// circular correlation will be real.
//
// the derivatives are:
//     d(a \star b) = (da) \star b + a * (db)   [* = circ conv]
//     d(a * b) = b \star (da) + a \star (db)
template<class MyDevice>
void CircularCorrelation::forward_dev_impl(
    const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  // Eigen's FFT doesn't seem to work on GPU (although it does compile), the
  // TF implementation of FFT uses Eigen only on CPU and calls cuFFT for the
  // GPU impl. We should do something similar here to support GPU.
  // See
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/fft_ops.cc
  DYNET_NO_CUDA_IMPL_ERROR("CircularCorrelation forward");
#else
  // TODO implement batched version of this
  auto a = t<1>(*xs[0]);
  auto b = t<1>(*xs[1]);
  auto y = t<1>(fx);

  // set up memory for FFTs
  std::complex<float>* a_fft_mem = static_cast<std::complex<float>*>(aux_mem);
  std::complex<float>* b_fft_mem = a_fft_mem + xs[0]->d.size();
  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      a_fft(a_fft_mem, xs[0]->d.size());
  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      b_fft(b_fft_mem, xs[1]->d.size());

  // do FFTs
  const Eigen::array<ptrdiff_t, 1> fft {0};
  a_fft.device(*dev.edevice) =
      a.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);
  b_fft.device(*dev.edevice) =
      b.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);

  // this is circular correlation:
  auto ab_fft = a_fft.conjugate() * b_fft;
  y.device(*dev.edevice) =
      ab_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(fft);
#endif
}

template<class MyDevice>
void CircularCorrelation::backward_dev_impl(const MyDevice & dev,
                                            const vector<const Tensor*>& xs,
                                            const Tensor& fx,
                                            const Tensor& dEdf,
                                            unsigned i,
                                            Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("CircularCorrelation backward");
#else
  // grab the results of the FFTs of xs[0] and xs[1] that were computed
  // during forward and are needed here.
  std::complex<float>* a_fft_mem = static_cast<std::complex<float>*>(aux_mem);
  std::complex<float>* b_fft_mem = a_fft_mem + xs[0]->d.size();
  const Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      a_fft(a_fft_mem, xs[0]->d.size());
  const Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      b_fft(b_fft_mem, xs[1]->d.size());

  // Normally we would just add the value of the derivative directly to dEdxi,
  // but Eigen requires that the evaluation of an FFT be assigned somewhere, so
  // we allocate some scratch memory and wrap it with dtmp:
  AlignedMemoryPool* scratch_allocator =
      fx.device->pools[(int)DeviceMempool::SCS];
  float* tmpmem = static_cast<float*>(scratch_allocator->allocate(
      dEdxi.d.size() * sizeof(float)));
  Eigen::TensorMap<Eigen::Tensor<float, 1>> dtmp(tmpmem, xs[i]->d.size());

  // we also need memory for the FFT of dedf
  std::complex<float>* dr_fft_mem =
      static_cast<std::complex<float>*>(scratch_allocator->allocate(
          dEdxi.d.size() * sizeof(std::complex<float>)));
  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      dr_fft(dr_fft_mem, xs[i]->d.size());
  auto dr = t<1>(dEdf);
  auto out = t<1>(dEdxi);
  // do FFT of dedf
  const Eigen::array<ptrdiff_t, 1> fft {0};
  dr_fft.device(*dev.edevice) =
      dr.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);
  if (i == 0) {
    // circ_corr(dr, b)
    auto d_fft = dr_fft.conjugate() * b_fft;
    dtmp.device(*dev.edevice) =
        d_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(fft);
  } else {
    // circ_conv(a, dr)
    auto d_fft = a_fft * dr_fft;
    dtmp.device(*dev.edevice) =
        d_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(fft);
  }
  out.device(*dev.edevice) += dtmp;
  scratch_allocator->free();
#endif
}
DYNET_NODE_INST_DEV_IMPL(CircularCorrelation)

// ************* CircularConvolution *************

#ifndef __CUDACC__

string CircularConvolution::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "circ_conv(" << arg_names[0] << ", " << arg_names[1] << ')';
  return s.str();
}

Dim CircularConvolution::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs[0].ndims() == 1 && xs[0] == xs[1],
                  "Bad input dimensions in CircularConvolution: " << xs);
  return xs[0];
}

size_t CircularConvolution::aux_storage_size() const {
  return dim.size() * sizeof(std::complex<float>) * 2;
}

#endif

template<class MyDevice>
void CircularConvolution::forward_dev_impl(
    const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("CircularConvolution forward");
#else
  auto a = t<1>(*xs[0]);
  auto b = t<1>(*xs[1]);
  auto y = t<1>(fx);

  // set up memory for FFTs
  std::complex<float>* a_fft_mem = static_cast<std::complex<float>*>(aux_mem);
  std::complex<float>* b_fft_mem = a_fft_mem + xs[0]->d.size();
  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      a_fft(a_fft_mem, xs[0]->d.size());
  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      b_fft(b_fft_mem, xs[1]->d.size());

  // do FFTs
  const Eigen::array<ptrdiff_t, 1> fft {0};
  a_fft.device(*dev.edevice) =
      a.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);
  b_fft.device(*dev.edevice) =
      b.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);

  // this is circular convolution:
  auto ab_fft = a_fft * b_fft;
  y.device(*dev.edevice) =
    ab_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(fft);
#endif
}

template<class MyDevice>
void CircularConvolution::backward_dev_impl(const MyDevice & dev,
                                            const vector<const Tensor*>& xs,
                                            const Tensor& fx,
                                            const Tensor& dEdf,
                                            unsigned i,
                                            Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("CircularConvolution backward");
#else
  // grab the results of the FFTs of xs[0] and xs[1] that were computed
  // during forward and are needed here.
  std::complex<float>* a_fft_mem = static_cast<std::complex<float>*>(aux_mem);
  std::complex<float>* b_fft_mem = a_fft_mem + xs[0]->d.size();
  const Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      a_fft(a_fft_mem, xs[0]->d.size());
  const Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      b_fft(b_fft_mem, xs[1]->d.size());

  // Normally we would just add the value of the derivative directly to dEdxi,
  // but Eigen requires that the evaluation of an FFT be assigned somewhere, so
  // we allocate some scratch memory and wrap it with dtmp:
  AlignedMemoryPool* scratch_allocator =
      fx.device->pools[(int)DeviceMempool::SCS];
  float* tmpmem = static_cast<float*>(scratch_allocator->allocate(
      dEdxi.d.size() * sizeof(float)));
  Eigen::TensorMap<Eigen::Tensor<float, 1>> dtmp(tmpmem, xs[i]->d.size());

  // we also need memory for the FFT of dedf
  std::complex<float>* dr_fft_mem =
      static_cast<std::complex<float>*>(scratch_allocator->allocate(
          dEdxi.d.size() * sizeof(std::complex<float>)));
  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1>>
      dr_fft(dr_fft_mem, xs[i]->d.size());
  auto dr = t<1>(dEdf);
  auto out = t<1>(dEdxi);
  // do FFT of dedf
  const Eigen::array<ptrdiff_t, 1> fft {0};
  dr_fft.device(*dev.edevice) =
      dr.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);
  if (i == 0) {
    // circ_cor(b, dr)
    auto d_fft = b_fft.conjugate() * dr_fft;
    dtmp.device(*dev.edevice) =
        d_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(fft);
  } else {
    // circ_cor(a, dr)
    auto d_fft = a_fft.conjugate() * dr_fft;
    dtmp.device(*dev.edevice) =
        d_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(fft);
  }
  out.device(*dev.edevice) += dtmp;
  scratch_allocator->free();
#endif
}
DYNET_NODE_INST_DEV_IMPL(CircularConvolution)

}  // namespace dynet
