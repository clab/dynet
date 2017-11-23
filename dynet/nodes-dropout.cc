#include "dynet/tensor-eigen.h"
#include "dynet/nodes-dropout.h"

#include "dynet/nodes-impl-macros.h"

using namespace std;

namespace dynet {

// ************* Dropout *************

#ifndef __CUDACC__

string Dropout::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dropout(" << arg_names[0] << ",p=" << p << ')';
  return s.str();
}

Dim Dropout::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Dropout")
  return xs[0];
}

size_t Dropout::aux_storage_size() const {
  return dim.size() * sizeof(float);
}

#endif

template<class MyDevice>
void Dropout::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Tensor m(dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  TensorTools::randomize_bernoulli(m, (1.f-p), 1.f / (1.f-p));
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]) * tvec(m);
}

template<class MyDevice>
void Dropout::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Tensor m(dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  tvec(dEdxi).device(*dev.edevice) += tvec(dEdf) * tvec(m);
}
DYNET_NODE_INST_DEV_IMPL(Dropout)

// ************* DropoutDim *************

#ifndef __CUDACC__

string DropoutDim::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dropout_dim(" << arg_names[0] << ",p=" << p << ')';
  return s.str();
}

Dim DropoutDim::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in DropoutDim")
  DYNET_ARG_CHECK(xs[0].nd < 4, "DropoutDim only supports tensor up to order 3 + batch dimension, got tensor of order"<<xs[0].nd)
  DYNET_ARG_CHECK(xs[0].nd > dimension, "In DropoutDim : tried to drop along dimension "<<dimension<<" on tensor of order"<<xs[0].nd)
  return xs[0];
}

size_t DropoutDim::aux_storage_size() const {
  return (dim.size() / dim[dimension]) * sizeof(float);
}

#endif

template<class MyDevice>
void DropoutDim::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Dim mask_dim(dim);
  mask_dim.d[dimension]=1;
  Tensor m(mask_dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  TensorTools::randomize_bernoulli(m, (1.f-p), 1.f / (1.f-p));
  Eigen::array<ptrdiff_t, 4> bcast = {1, 1, 1, 1}; bcast[dimension] = xs[0]->d[dimension];
  tb<3>(fx).device(*dev.edevice) = tb<3>(*xs[0]) * tb<3>(m).broadcast(bcast);
}

template<class MyDevice>
void DropoutDim::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Dim mask_dim(dim);
  mask_dim.d[dimension]=1;
  Tensor m(mask_dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> bcast = {1, 1, 1, 1}; bcast[dimension] = dEdf.d[dimension];
  tb<3>(dEdxi).device(*dev.edevice) += tb<3>(dEdf) * tb<3>(m).broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(DropoutDim)

// ************* DropoutBatch *************

#ifndef __CUDACC__

string DropoutBatch::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dropout_batch(" << arg_names[0] << ",p=" << p << ')';
  return s.str();
}

Dim DropoutBatch::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in DropoutBatch")
  return xs[0];
}

size_t DropoutBatch::aux_storage_size() const {
  return dim.batch_elems() * sizeof(float);
}

#endif

template<class MyDevice>
void DropoutBatch::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  Dim mask_dim({1},xs[0]->d.batch_elems());
  Tensor m(mask_dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  TensorTools::randomize_bernoulli(m, (1.f-p), 1.f / (1.f-p));
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  tbvec(fx).device(*dev.edevice) = tbvec(*xs[0]) * tbvec(m).broadcast(bcast);
}

template<class MyDevice>
void DropoutBatch::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  Dim mask_dim({1},xs[0]->d.batch_elems());
  Tensor m(mask_dim, (float*)aux_mem, fx.device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 2> bcast = {xs[0]->d.batch_size(), 1};
  tbvec(dEdxi).device(*dev.edevice) += tbvec(dEdf) * tbvec(m).broadcast(bcast);
}
DYNET_NODE_INST_DEV_IMPL(DropoutBatch)

// ************* BlockDropout *************

#ifndef __CUDACC__

string BlockDropout::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "block_dropout(" << arg_names[0] << ",dropout_probability=" << dropout_probability << ')';
  return s.str();
}

Dim BlockDropout::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in BlockDropout")
  return xs[0];
}

size_t BlockDropout::aux_storage_size() const {
  // we just need to remember whether this entire block is turned on (1.0) or off (0.0)
  return 1 * sizeof(float);
}

#endif

template<class MyDevice>
void BlockDropout::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  bernoulli_distribution distribution(1.0 - dropout_probability);
  float block_multiplier = distribution(*rndeng)? 1.0 : 0.0;
  block_multiplier =
    dropout_probability == 1.0? 0.0 : block_multiplier / (1.0 - dropout_probability);
  if (dropout_probability > 1.0 || dropout_probability < 0.0)
    DYNET_INVALID_ARG("Dropout probability must be in the range [0, 1]");
  *(static_cast<float*>(aux_mem)) = block_multiplier;
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]) * block_multiplier;
}

template<class MyDevice>
void BlockDropout::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  float block_multiplier = *(static_cast<float*>(aux_mem));
  tvec(dEdxi).device(*dev.edevice) += tvec(dEdf) * block_multiplier;
}
DYNET_NODE_INST_DEV_IMPL(BlockDropout)

}
