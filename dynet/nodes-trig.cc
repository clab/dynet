#include "dynet/tensor-eigen.h"
#include "dynet/nodes-trig.h"

#include "dynet/nodes-impl-macros.h"
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
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).tanh();
}

template<class MyDevice>
void Tanh::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(fx).binaryExpr(tvec(dEdf), scalar_tanh_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Tanh)

}
