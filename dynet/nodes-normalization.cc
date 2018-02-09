#include "dynet/tensor-eigen.h"
#include "dynet/nodes-normalization.h"

#include "dynet/nodes-impl-macros.h"

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
  AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];
  Tensor tmp(Dim({1}, 1), nullptr, fx.device, fx.mem_pool);
  tmp.v = static_cast<float*>(scratch_allocator->allocate(tmp.d.size() * sizeof(float)));
  tvec(tmp).device(*dev.edevice) = tvec(*xs[0]).square().sum(red_axis).sqrt().reshape(morph);
  tvec(fx).device(*dev.edevice) = (tvec(*xs[0]) / tvec(tmp).broadcast(bcast)) * as_scalar(*xs[1]);
  scratch_allocator->free();
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
  AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];
  if (i==0){
    Tensor tmp(Dim({1}, 1), nullptr, fx.device, fx.mem_pool);
    tmp.v = static_cast<float*>(scratch_allocator->allocate(tmp.d.size() * sizeof(float)));
    tvec(tmp).device(*dev.edevice) = tvec(*xs[0]).square().sum(red_axis).sqrt().reshape(morph);
    Tensor tmp2(Dim({1}, 1), nullptr, fx.device, fx.mem_pool);
    tmp2.v = static_cast<float*>(scratch_allocator->allocate(tmp2.d.size() * sizeof(float)));
    tvec(tmp2).device(*dev.edevice) = (((tvec(dEdf) * tvec(*xs[0])).sum(red_axis)) / tvec(*xs[0]).square().sum(red_axis)).reshape(morph);
    tvec(dEdxi).device(*dev.edevice) += (tvec(dEdf) / tvec(tmp).broadcast(bcast)) * as_scalar(*xs[1]) - tvec(fx) * tvec(tmp2).broadcast(bcast);
  }else{
    t<0>(dEdxi).device(*dev.edevice) += ((tvec(dEdf) * tvec(*xs[0])).sum(red_axis)) /  tvec(*xs[0]).square().sum(red_axis).sqrt();
  }
  scratch_allocator->free();
}
DYNET_NODE_INST_DEV_IMPL(WeightNormalization)

}
