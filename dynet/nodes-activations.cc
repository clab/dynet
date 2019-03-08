#include "dynet/tensor-eigen.h"
#include "dynet/nodes-activations.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"

#include "dynet/simd-functors.h"

using namespace std;

namespace dynet {

// ************* Rectify *************

#ifndef __CUDACC__

string Rectify::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "ReLU(" << arg_names[0] << ')';
  return s.str();
}

Dim Rectify::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Rectify");
  return xs[0];
}

#endif

template<class MyDevice>
void Rectify::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in Rectify::forward");
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).cwiseMax(0.f);
}

template<class MyDevice>
void Rectify::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(fx).cast<bool>().cast<float>() * tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(Rectify)

// ************* LogisticSigmoid *************

#ifndef __CUDACC__

string LogisticSigmoid::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "\\sigma(" << arg_names[0] << ')';
  return s.str();
}

Dim LogisticSigmoid::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in LogisticSigmoid")
  return xs[0];
}

#endif

template<class MyDevice>
void LogisticSigmoid::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in LogisticSigmoid::forward");
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).unaryExpr(scalar_logistic_sigmoid_op<float>());
}

template<class MyDevice>
void LogisticSigmoid::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(fx).binaryExpr(tvec(dEdf), scalar_logistic_sigmoid_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(LogisticSigmoid)

// ************* SoftSign *************

#ifndef __CUDACC__

string SoftSign::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "softsign(" << arg_names[0] << ')';
  return s.str();
}

Dim SoftSign::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in SoftSign");
  DYNET_ARG_CHECK(LooksLikeVector(xs[0]), "Bad input dimensions in SoftSign: " << xs);
  return xs[0];
}

#endif

template<class MyDevice>
void SoftSign::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in SoftSign::forward");
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).unaryExpr(FSoftSign());
}

template<class MyDevice>
void SoftSign::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(fx).binaryExpr(tvec(dEdf), FSoftSignBackward());
}
DYNET_NODE_INST_DEV_IMPL(SoftSign)

// ************* Erf *************

#ifndef __CUDACC__

string Erf::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "erf(" << arg_names[0] << ')';
  return s.str();
}

Dim Erf::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Erf")
  return xs[0];
}

#endif

template<class MyDevice>
void Erf::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).erf();
}

template<class MyDevice>
void Erf::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(*xs[0]).binaryExpr(tvec(dEdf), scalar_erf_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Erf)

// ************* ExponentialLinearUnit *************

#ifndef __CUDACC__

string ExponentialLinearUnit::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "ELU(" << arg_names[0] << ", lambda=" << lambda << ", alpha=" << alpha << ')';
  return s.str();
}

Dim ExponentialLinearUnit::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in ExponentialLinearUnit");
  return xs[0];
}

#endif

template<class MyDevice>
void ExponentialLinearUnit::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed dimension check in ExponentialLinearUnit::forward");
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).unaryExpr(FELUForward(alpha, lambda));;
}

template<class MyDevice>
void ExponentialLinearUnit::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(*xs[0]).binaryExpr(tvec(dEdf), FELUBackward(alpha, lambda));
}
DYNET_NODE_INST_DEV_IMPL(ExponentialLinearUnit)

// ************* SigmoidLinearUnit *************

#ifndef __CUDACC__

string SigmoidLinearUnit::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << "*\\sigma(" << arg_names[0] << "*beta), beta=" << beta << ')';
  return s.str();
}

Dim SigmoidLinearUnit::dim_forward(const vector<Dim>& xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed input count check in SigmoidLinearUnit")
  return xs[0];
}

#endif

template<class MyDevice>
void SigmoidLinearUnit::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in SigmoidLinearUnit::forward");
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).unaryExpr(FSILUForward(beta));;
}

template<class MyDevice>
void SigmoidLinearUnit::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(*xs[0]).binaryExpr(tvec(dEdf), FSILUBackward(beta));
}
DYNET_NODE_INST_DEV_IMPL(SigmoidLinearUnit)

}

