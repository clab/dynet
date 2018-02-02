#include "dynet/tensor-eigen.h"
#include "dynet/nodes-argmax.h"

#include "dynet/nodes-impl-macros.h"


#include "dynet/tensor.h"
#include "dynet/index-tensor.h"

#ifdef __CUDACC__
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#endif

using namespace std;

namespace dynet {

// ************* Argmax *************

#ifndef __CUDACC__

string Argmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << (straight_through ? "straight_through(" : "argmax(") << arg_names[0] << ")_{" << dim << '}';
  return s.str();
}

Dim Argmax::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Argmax");
  // For now only support 1 dim
  DYNET_ARG_CHECK(xs[0].nd == 1, "Argmax only supports vectors for now, got dimension " << xs);
  DYNET_ARG_CHECK(d < xs[0].nd, "Cannot compute argmax along dimension " << dim << " for tensor of shape " << xs);
  return xs[0];
}

size_t Argmax::aux_storage_size() const {
  return dim.size() / dim[d] * sizeof(Eigen::DenseIndex);
}

#endif

template<class MyDevice>
void Argmax::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Eigen::DenseIndex* argmax_ids_mem = static_cast<Eigen::DenseIndex*>(aux_mem);
  Dim argmax_dim({1}, xs[0]->d.bd);
  IndexTensor argmax_ids(argmax_dim, argmax_ids_mem, fx.device, DeviceMempool::SCS);
  tb<0>(argmax_ids).device(*dev.edevice) = tb<1>(*xs[0]).argmax(d);
  std::vector<Eigen::DenseIndex> ids_vec = as_vector(argmax_ids);
  tvec(fx).device(*dev.edevice) = tvec(fx).constant(0.0);
  for (unsigned b=0; b<xs[0]->d.bd; b++){
      int idx = ids_vec[b] + b * (xs[0]->d[d]);
      TensorTools::set_element(fx, idx, 1.0);
  }
}

template<class MyDevice>
void Argmax::backward_dev_impl(const MyDevice & dev,
                            const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  // If we're using the straight-through estimator: copy gradient
  if (straight_through)
    tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
  // Otherwise no gradient!
}
DYNET_NODE_INST_DEV_IMPL(Argmax)

}
