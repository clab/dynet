#include "dynet/tensor-eigen.h"
#include "dynet/nodes-flow.h"

#include "dynet/nodes-impl-macros.h"

using namespace std;

namespace dynet {

// ************* Reshape *************

#ifndef __CUDACC__

string Reshape::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "reshape(" << arg_names[0] << " --> " << to << ')';
  return s.str();
}

Dim Reshape::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Reshape")
  if(to.size() == xs[0].size()) {
    return to;
  } else {
    DYNET_ARG_CHECK(to.batch_elems() == 1 && to.batch_size() == xs[0].batch_size(),
                    "Bad arguments to Reshape: " << to << ", " << xs[0]);
    Dim ret(to);
    ret.bd = xs[0].batch_elems();
    return ret;
  }
}

#endif

template<class MyDevice>
void Reshape::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  // just point to the input memory and change dimensions
  // dimensions are handled by forward_dim
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
}

template<class MyDevice>
void Reshape::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  const Tensor reshaped(dEdxi.d, dEdf.v, dEdxi.device, dEdf.mem_pool);
  tvec(dEdxi).device(*dev.edevice) += tvec(reshaped);
}
DYNET_NODE_INST_DEV_IMPL(Reshape)

// ************* Identity *************

#ifndef __CUDACC__

string Identity::as_string(const vector<string>& arg_names) const {
  return arg_names[0];
}

Dim Identity::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Identity")
  return xs[0];
}

#endif

template<class MyDevice>
void Identity::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
}

template<class MyDevice>
void Identity::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(Identity)

// ************* NoBackprop *************

#ifndef __CUDACC__

string NoBackprop::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "nobackprop(" << arg_names[0] << ')';
  return s.str();
}

Dim NoBackprop::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in NoBackprop")
  return xs[0];
}

#endif

template<class MyDevice>
void NoBackprop::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
}

template<class MyDevice>
void NoBackprop::backward_dev_impl(const MyDevice & dev,
                                   const vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {
  // no op
}
DYNET_NODE_INST_DEV_IMPL(NoBackprop)

// ************* ScaleGradient *************

#ifndef __CUDACC__

string ScaleGradient::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "flip_gradient(" << arg_names[0] << ')';
  return s.str();
}

Dim ScaleGradient::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in ScaleGradient");
  return xs[0];
}

#endif

template<class MyDevice>
void ScaleGradient::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
}

template<class MyDevice>
void ScaleGradient::backward_dev_impl(const MyDevice & dev,
                                   const vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {
  // Scale gradient by lambda on backprop
  tvec(dEdxi).device(*dev.edevice) += tvec(dEdf) * lambd;
}
DYNET_NODE_INST_DEV_IMPL(ScaleGradient)

}
