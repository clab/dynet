#include "dynet/nodes-flow.h"

#include "dynet/nodes-macros.h"

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
  fx.tvec().device(*dev.edevice) = xs[0]->tvec();
}

template<class MyDevice>
void Reshape::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  const Tensor reshaped(dEdxi.d, dEdf.v, dEdxi.device, dEdf.mem_pool);
  dEdxi.tvec().device(*dev.edevice) += reshaped.tvec();
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
  fx.tvec().device(*dev.edevice) = xs[0]->tvec();
}

template<class MyDevice>
void Identity::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
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
  fx.tvec().device(*dev.edevice) = xs[0]->tvec();
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

// ************* FlipGradient *************

#ifndef __CUDACC__

string FlipGradient::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "flip_gradient(" << arg_names[0] << ')';
  return s.str();
}

Dim FlipGradient::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in FlipGradient");
  return xs[0];
}

#endif

template<class MyDevice>
void FlipGradient::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec();
}

template<class MyDevice>
void FlipGradient::backward_dev_impl(const MyDevice & dev,
                                   const vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const {
  // takes negative on backprop
  dEdxi.tvec().device(*dev.edevice) -= dEdf.tvec();
}
DYNET_NODE_INST_DEV_IMPL(FlipGradient)

}
