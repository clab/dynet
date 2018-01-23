#include "dynet/tensor-eigen.h"
#include "dynet/nodes-arith-const.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"
#include "dynet/simd-functors.h"

using namespace std;

namespace dynet {

// ************* ConstantMinusX *************

#ifndef __CUDACC__

string ConstantMinusX::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << c << " - " << arg_names[0];
  return s.str();
}

Dim ConstantMinusX::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in ConstantMinusX")
  return xs[0];
}

#endif

template<class MyDevice>
void ConstantMinusX::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).unaryExpr(const_minus_op<float>(c));
}

template<class MyDevice>
void ConstantMinusX::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) -= tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(ConstantMinusX)

// ************* ConstantPlusX *************

#ifndef __CUDACC__

string ConstantPlusX::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << c << " + " << arg_names[0];
  return s.str();
}

Dim ConstantPlusX::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in ConstantPlusX")
  return xs[0];
}

#endif

template<class MyDevice>
void ConstantPlusX::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).unaryExpr(const_add_op<float>(c));
}

template<class MyDevice>
void ConstantPlusX::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(ConstantPlusX)

// ************* ConstScalarMultiply *************

#ifndef __CUDACC__

string ConstScalarMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << alpha;
  return s.str();
}

Dim ConstScalarMultiply::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "ConstScalarMultiply expects one argument: " << xs);
  return xs[0];
}

#endif

template<class MyDevice>
void ConstScalarMultiply::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]) * alpha;
}

template<class MyDevice>
void ConstScalarMultiply::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i == 0, "Failed dimension check in ConstScalarMultiply");
  tvec(dEdxi).device(*dev.edevice) += tvec(dEdf) * alpha;
}
DYNET_NODE_INST_DEV_IMPL(ConstScalarMultiply)

}
