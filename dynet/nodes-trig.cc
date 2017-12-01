#include "dynet/nodes-trig.h"

#include "dynet/nodes-macros.h"
#include "dynet/simd-functors.h"

using namespace std;

namespace dynet {

// ************* Tanh *************

#ifndef __CUDACC__

string Tanh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "tanh(" << arg_names[0] << ')';
  return s.str();
}

Dim Tanh::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Tanh")
  return xs[0];
}

#endif

template<class MyDevice>
void Tanh::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().tanh();
}

template<class MyDevice>
void Tanh::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), scalar_tanh_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Tanh)

// ************* Acosh *************

#ifndef __CUDACC__

string Acosh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "acosh(" << arg_names[0] << ')';
  return s.str();
}

Dim Acosh::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Acosh")
  return xs[0];
}

#endif

template<class MyDevice>
void Acosh::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().unaryExpr(scalar_acosh_forward_op<float>());
}

template<class MyDevice>
void Acosh::backward_dev_impl(const MyDevice & dev,
                              const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += xs[0]->tvec().binaryExpr(dEdf.tvec(), scalar_acosh_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Acosh)

}  // namespace dynet
