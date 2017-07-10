#include "dynet/nodes-activations.h"

#include "dynet/nodes-macros.h"
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
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().cwiseMax(0.f);
}

template<class MyDevice>
void Rectify::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), FRectifyBackward());
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
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().unaryExpr(scalar_logistic_sigmoid_op<float>());
}

template<class MyDevice>
void LogisticSigmoid::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), scalar_logistic_sigmoid_backward_op<float>());
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
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().unaryExpr(FSoftSign());
}

template<class MyDevice>
void SoftSign::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), FSoftSignBackward());
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
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().erf();
}

template<class MyDevice>
void Erf::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += xs[0]->tvec().binaryExpr(dEdf.tvec(), scalar_erf_backward_op<float>());
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
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().unaryExpr(FELUForward(alpha, lambda));;
}

template<class MyDevice>
void ExponentialLinearUnit::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += xs[0]->tvec().binaryExpr(dEdf.tvec(), FELUBackward(alpha, lambda));
}
DYNET_NODE_INST_DEV_IMPL(ExponentialLinearUnit)

}
