#include "dynet/nodes-normalization.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* WeightNormalization *************

#ifndef __CUDACC__

string WeightNormalization::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "weight_norm(" << arg_names[0] << ", " << arg_names[1] << ')';
  return s.str();
}

Dim WeightNormalization::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in WeightNormalization");
  DYNET_ARG_CHECK(1 == xs[1].size()," Size of gain parameter in WeightNormalization should be 1, received " << xs[1].size());
  return xs[0];
}

#endif

template<class MyDevice>
void WeightNormalization::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in WeightNormalization::forward");
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  Eigen::array<ptrdiff_t, 1> bcast = {xs[0]->d.size()};
  Eigen::array<ptrdiff_t, 1> morph = {1};
  fx.tvec().device(*dev.edevice) = (xs[0]->tvec() / xs[0]->tvec().square().sum(red_axis).sqrt().reshape(morph).broadcast(bcast)) * as_scalar(*xs[1]);
}

template<class MyDevice>
void WeightNormalization::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Eigen::array<ptrdiff_t, 1> red_axis = {0};
  Eigen::array<ptrdiff_t, 1> bcast = {xs[0]->d.size()};
  Eigen::array<ptrdiff_t, 1> morph = {1};
  if (i==0){
    dEdxi.tvec().device(*dev.edevice) += (dEdf.tvec() / xs[0]->tvec().square().sum(red_axis).sqrt().reshape(morph).broadcast(bcast)) * as_scalar(*xs[1]) - fx.tvec() * (((dEdf.tvec() * xs[0]->tvec()).sum(red_axis)) / xs[0]->tvec().square().sum(red_axis)).reshape(morph).broadcast(bcast);
  }else{
    dEdxi.t<0>().device(*dev.edevice) += ((dEdf.tvec() * xs[0]->tvec()).sum(red_axis)) /  xs[0]->tvec().square().sum(red_axis).sqrt();
  }
}
DYNET_NODE_INST_DEV_IMPL(WeightNormalization)

}
