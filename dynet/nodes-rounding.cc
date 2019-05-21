#include "dynet/tensor-eigen.h"
#include "dynet/nodes-rounding.h"

#include "dynet/nodes-impl-macros.h"

using namespace std;

namespace dynet {

// ************* Round *************

#ifndef __CUDACC__

string Round::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "round(" << arg_names[0] << ')';
  return s.str();
}

Dim Round::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Round")
  return xs[0];
}

#endif

template<class MyDevice>
void Round::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).round();
}

template<class MyDevice>
void Round::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if (straight_through)
    tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(Round)

// ************* Ceil *************

#ifndef __CUDACC__

string Ceil::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "ceil(" << arg_names[0] << ')';
  return s.str();
}

Dim Ceil::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Ceil")
  return xs[0];
}

#endif

template<class MyDevice>
void Ceil::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).ceil();
}

template<class MyDevice>
void Ceil::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if (straight_through)
    tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(Ceil)

// ************* Floor *************

#ifndef __CUDACC__

string Floor::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "floor(" << arg_names[0] << ')';
  return s.str();
}

Dim Floor::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Floor")
  return xs[0];
}

#endif

template<class MyDevice>
void Floor::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).floor();
}

template<class MyDevice>
void Floor::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if (straight_through)
    tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(Floor)

}
