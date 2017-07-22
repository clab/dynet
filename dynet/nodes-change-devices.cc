#include <vector>
#include <string>

#include "dynet/nodes-change-devices.h"
#include "dynet/tensor.h"
#include "dynet/dim.h"
#include "dynet/nodes-macros.h"
#include "dynet/functors.h"

using namespace std;

namespace dynet {

#ifndef __CUDACC__
string ToDevice::as_string(const vector<string>& arg_names) const {
  return "copy tensor between devices";
}

Dim ToDevice::dim_forward(const vector<Dim> & xs) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in ToDevice::dim_forward");
  return xs[0];
}
#endif

template<class MyDevice>
void ToDevice::forward_dev_impl(const MyDevice & dev,
                                const vector<const Tensor*> & xs,
                                Tensor & fx) const {
#ifdef __CUDACC__
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in ToDevice::forward");
  fx.d = xs[0]->d;
  fx.device = xs[0]->device;
  fx.mem_pool = DeviceMempool::FXS;
  fx.v = static_cast<float*>(fx.device->pools[(int)DeviceMempool::FXS]
                             ->allocate(fx.d.size() * sizeof(float)));
  TensorTools::copy_elements(fx, *xs[0]);
#endif
}

template<class MyDevice>
void ToDevice::backward_dev_impl(const MyDevice & dev,
                                 const vector<const Tensor*> & xs,
                                 const Tensor & fx,
                                 const Tensor & dEdf,
                                 unsigned i,
                                 Tensor & dEdxi) const {
#ifdef __CUDACC__
  dEdxi.d = fx.d;
  dEdxi.device = fx.device;
  dEdxi.mem_pool = DeviceMempool::DEDFS;
  dEdxi.v = static_cast<float*>(dEdxi.device->pools[(int)DeviceMempool::DEDFS]
                                ->allocate(dEdxi.d.size() * sizeof(float)));
  TensorTools::copy_elements(dEdxi, fx);
#endif
}
DYNET_NODE_INST_DEV_IMPL(ToDevice)

} // namespace dynet
