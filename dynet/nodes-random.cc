#include "dynet/tensor-eigen.h"
#include "dynet/nodes-random.h"

#include "dynet/nodes-impl-macros.h"

using namespace std;

namespace dynet {

// ************* GaussianNoise *************

#ifndef __CUDACC__

string GaussianNoise::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " + N(0," << stddev << ')';
  return s.str();
}

Dim GaussianNoise::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in GaussianNoise")
  return xs[0];
}

#endif

template<class MyDevice>
void GaussianNoise::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {

  AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];
  Tensor noise(dim, nullptr, fx.device, fx.mem_pool);
  noise.v = static_cast<float*>(scratch_allocator->allocate(noise.d.size() * sizeof(float)));
  TensorTools::randomize_normal(noise, 0, stddev);

  tvec(fx).device(*dev.edevice) = tvec(*xs[0]) + tvec(noise);

  scratch_allocator->free();
}

template<class MyDevice>
void GaussianNoise::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(GaussianNoise)

// ************* RandomNormal *************

#ifndef __CUDACC__

string RandomNormal::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "random_normal(" << dim << ')';
  return s.str();
}

Dim RandomNormal::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

#endif

template<class MyDevice>
void RandomNormal::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in RandomNormal::forward");
  TensorTools::randomize_normal(fx, mean, stddev);
}

template<class MyDevice>
void RandomNormal::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(RandomNormal)

// ************* RandomBernoulli *************

#ifndef __CUDACC__

string RandomBernoulli::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "random_bernoulli(" << dim << ", " << p << ')';
  return s.str();
}

Dim RandomBernoulli::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

#endif

template<class MyDevice>
void RandomBernoulli::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in RandomBernoulli::forward");
  TensorTools::randomize_bernoulli(fx, p, scale);
}

template<class MyDevice>
void RandomBernoulli::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(RandomBernoulli)

// ************* RandomUniform *************

#ifndef __CUDACC__

string RandomUniform::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "random_uniform(" << dim << ", " << left << ", " << right << ')';
  return s.str();
}

Dim RandomUniform::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

#endif

template<class MyDevice>
void RandomUniform::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in RandomUniform::forward");
  TensorTools::randomize_uniform(fx, left, right);
}

template<class MyDevice>
void RandomUniform::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(RandomUniform)

// ************* RandomGumbel *************

#ifndef __CUDACC__

string RandomGumbel::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "random_gumbel(" << dim << ", " << mu << ", " << beta << ')';
  return s.str();
}

Dim RandomGumbel::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

#endif

template<class MyDevice>
void RandomGumbel::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in RandomGumbel::forward");
  DYNET_ARG_CHECK(mu == 0.0 && beta == 1.0, "RandomGumbel only supports Gumbel(0,1) at the moment (pull requests welcome)");
  TensorTools::randomize_uniform(fx, 0, 1);
  float eps = 1e-20;
  tvec(fx).device(*dev.edevice) = -(-tvec(fx).cwiseMax(eps).log()).cwiseMax(eps).log();
}

template<class MyDevice>
void RandomGumbel::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(RandomGumbel)


}
