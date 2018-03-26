#include "dynet/tensor-eigen.h"
#include "dynet/nodes-linalg.h"
#include "dynet/nodes-impl-macros.h"
#include "dynet/except.h"

using namespace std;

namespace dynet {

// ************* Transpose *************

#ifndef __CUDACC__

string Transpose::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "transpose("<< arg_names[0] << ", ";
  for(size_t i = 0; i < dims.size(); ++i)
    s << (i == 0?'{':',') << dims[i];
  s << "})";
  return s.str();
}

Dim Transpose::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Bad arguments to Transpose: " << xs);
  DYNET_ARG_CHECK(xs[0].nd == dims.size() || xs[0].num_nonone_dims() == 1, "Dimensions passed to transpose (" << dims.size() << ") must be equal to dimensions in input tensor (" << xs[0].nd << ')');
  Dim ret(xs[0]);
  ret.nd = dims.size();
  for(size_t i = 0; i < dims.size(); ++i)
    ret.d[i] = xs[0][dims[i]];
  return ret;
}


int Transpose::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  Sig s(nt::transpose);
  s.add_dim(cg.nodes[args[0]]->dim);
  return sm.get_idx(s);
}

std::vector<int> Transpose::autobatch_concat(const ComputationGraph & cg) const {
  return vector<int>(1,1);
}

#endif

template<class MyDevice>
void Transpose::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  if (dim.num_nonone_dims() <= 1) {
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
  } else {
    Eigen::array<ptrdiff_t, 5> order;
    for(size_t i = 0; i < 5; ++i)
      order[i] = (i >= dims.size() ? i : dims[i]);
    tb<4>(fx).device(*dev.edevice) = tb<4>(*xs[0]).shuffle(order);
  }
}

template<class MyDevice>
void Transpose::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Eigen::array<ptrdiff_t, 5> order;
  for(size_t i = 0; i < 5; ++i)
    order[(i >= dims.size() ? i : dims[i])] = i;
  tb<4>(dEdxi).device(*dev.edevice) += tb<4>(dEdf).shuffle(order);
}
DYNET_NODE_INST_DEV_IMPL(Transpose)

// ************* MatrixInverse *************

#ifndef __CUDACC__

string MatrixInverse::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "inverse(" << arg_names[0] << ")";
  return s.str();
}

Dim MatrixInverse::dim_forward(const vector<Dim>& xs) const {
  return xs[0];
}

#endif

template<class MyDevice>
void MatrixInverse::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in MatrixInverse::forward");
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("MatrixInverse forward");
#else
  auto x = mat(*xs[0]);
  auto y = mat(fx);
  y = x.inverse();
  // TODO: Change into tensors after resolving test errors
  //t<2>(fx).device(*dev.edevice) = t<2>(*xs[0]).inverse();
#endif
}

template<class MyDevice>
void MatrixInverse::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in MatrixInverse::backward");
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("MatrixInverse backward");
#else
  auto d = mat(dEdf);
  auto y = mat(fx);
  (mat(dEdxi)).noalias() -= y.transpose() * d * y.transpose();
#endif
}
DYNET_NODE_INST_DEV_IMPL(MatrixInverse)

// ************* LogDet *************

#ifndef __CUDACC__

string LogDet::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "logdet(" << arg_names[0] << ")";
  return s.str();
}

Dim LogDet::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs[0].ndims() <= 2 && (xs[0].rows() == xs[0].cols()), "Bad arguments in LogDet: " << xs);
  return Dim({1});
}

// set use_cholesky if M is symmetric - it's faster and more stable
// for dep parsing it won't be
template <typename MatrixType>
inline typename MatrixType::Scalar logdet(const MatrixType& M, bool use_cholesky = false) {
  using namespace Eigen;
  using std::log;
  typedef typename MatrixType::Scalar Scalar;
  Scalar ld = 0;
  if (use_cholesky) {
    LLT<Matrix<Scalar,Dynamic,Dynamic>> chol(M);
    auto& U = chol.matrixL();
    for (unsigned i = 0; i < M.rows(); ++i)
      ld += log(U(i,i));
    ld *= 2;
  } else {
    PartialPivLU<Matrix<Scalar,Dynamic,Dynamic>> lu(M);
    auto& LU = lu.matrixLU();
    Scalar c = lu.permutationP().determinant(); // -1 or 1
    for (unsigned i = 0; i < LU.rows(); ++i) {
      const auto& lii = LU(i,i);
      if (lii < Scalar(0)) c *= -1;
      ld += log(abs(lii));
    }
    ld += log(c);
  }
  return ld;
}

#endif

template<class MyDevice>
void LogDet::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("LogDet forward");
#else
  fx.v[0] = logdet(mat(*xs[0]), false);
#endif
}

template<class MyDevice>
void LogDet::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("LogDet backward");
#else
  auto trans = (mat(*xs[0])).transpose();
  (mat(dEdxi)) += (dEdf.v[0]) * trans.inverse();
#endif
}
DYNET_NODE_INST_DEV_IMPL(LogDet)

// ************* TraceOfProduct *************

#ifndef __CUDACC__

string TraceOfProduct::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "Tr(" << arg_names[0] << " * " << arg_names[1] << "^T)";
  return s.str();
}

Dim TraceOfProduct::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2 && xs[0] == xs[1], "Bad arguments in TraceOfProduct: " << xs);
  return Dim({1}, max(xs[0].bd, xs[1].bd));
}

#endif

template<class MyDevice>
void TraceOfProduct::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("TraceOfProduct forward");
#else
  auto x1 = mat(*xs[0]);
  auto x2 = mat(*xs[1]);
  fx.v[0] = (x1 * x2.transpose()).trace();
#endif
}

template<class MyDevice>
void TraceOfProduct::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ARG_CHECK(i < 2, "Failed dimension check in TraceOfProduce::backward");
#ifdef __CUDACC__
  DYNET_NO_CUDA_IMPL_ERROR("TraceOfProduct backward");
#else
  const float d = dEdf.v[0];
  auto xother = mat(*xs[1 - i]);
  mat(dEdxi) += d * xother;
#endif
}
DYNET_NODE_INST_DEV_IMPL(TraceOfProduct)

}
