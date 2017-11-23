#include "dynet/tensor-eigen.h"
#include <vector>
#include <string>

#include "dynet/nodes-to-device.h"
#include "dynet/tensor.h"
#include "dynet/dim.h"
#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"

using namespace std;

namespace dynet {

#ifndef __CUDACC__
string ToDevice::as_string(const vector<string>& arg_names) const {
  return "copy " + arg_names[0] + " between devices";
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
#ifdef HAVE_CUDA
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in ToDevice::forward");
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
#ifdef HAVE_CUDA
  Tensor tmp;
  tmp.d = dEdxi.d;
  tmp.device = dEdxi.device;
  tmp.mem_pool = dEdxi.mem_pool;
  tmp.v = static_cast<float*>(dEdxi.device->pools[(int)dEdxi.mem_pool]
                              ->allocate(dEdxi.d.size() * sizeof(float)));
  if (!tmp.v) DYNET_RUNTIME_ERR("out of memory while attempting to allocate space for aggregate_elements.");
  TensorTools::copy_elements(tmp, dEdf);
  TensorTools::accumulate(dEdxi, tmp);
#endif
}
DYNET_NODE_INST_DEV_IMPL(ToDevice)

} // namespace dynet
