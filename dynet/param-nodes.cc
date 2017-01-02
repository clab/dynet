#include "dynet/param-nodes.h"

#include <limits>
#include <cmath>
#include <stdexcept>

#include "dynet/nodes-macros.h"
#include "dynet/weight-decay.h"

#ifdef HAVE_CUDA
#include "dynet/gpu-ops.h"
#endif

using namespace std;

namespace dynet {

#ifndef __CUDACC__

string ConstParameterNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "const_parameters(" << dim << ") @ " << params.get();
  return s.str();
}

Dim ConstParameterNode::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 0);
  return dim;
}

string ParameterNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "parameters(" << dim << ") @ " << params.get();
  return s.str();
}

Dim ParameterNode::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 0);
  return dim;
}

void ParameterNode::accumulate_grad(const Tensor& g) {
  params.get()->accumulate_grad(g);
}

string InputNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "constant(" << dim << ')';
  return s.str();
}

Dim InputNode::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

string SparseInputNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sparse_constant(" << dim << ')';
  return s.str();
}

Dim SparseInputNode::dim_forward(const vector<Dim>& xs) const {
  assert(ids.size() == data.size());
  return dim;
}

size_t SparseInputNode::aux_storage_size() const {
  return ids.size() * (sizeof(float) + sizeof(unsigned int));
}

string ScalarInputNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "scalar_constant(" << pdata << ')';
  return s.str();
}

Dim ScalarInputNode::dim_forward(const vector<Dim>& xs) const {
  return Dim({1});
}

size_t LookupNode::aux_storage_size() const {
  return dim.bd * sizeof(unsigned);
}

string LookupNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "lookup_parameters(|x|=" << params.get()->values.size() << " --> " << dim << ") @ " << params.get();
  return s.str();
}

Dim LookupNode::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

// TODO: This should be made more efficient on GPU
void LookupNode::accumulate_grad(const Tensor& g) {
  if(pindex) {
    params.get()->accumulate_grad(*pindex, g);
  } else {
    assert (pindices);
    const vector<Tensor>& gb = g.batch_elems();
    for (unsigned b = 0; b < pindices->size(); ++b) {
      unsigned i = pindices->at(b);
      assert (i < params.get()->values.size());
      params.get()->accumulate_grad(i, gb[b]);
    }
  }
}

size_t LookupSequenceNode::aux_storage_size() const {
  return (dim.bd + dim[dim.nd-1]) * sizeof(unsigned);
}

string LookupSequenceNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "lookup_seq_parameters(|x|=" << params.get()->values.size() << " --> " << dim << ") @ " << params.get();
  return s.str();
}

void LookupSequenceNode::create_dim() {
  size_t max_len = 0;
  if(pindex) {
    max_len = pindex->size();
  } else {
    assert(pindices);
    for(auto & p : *pindices)
      max_len = max(max_len, p.size());
    dim.bd = pindices->size();
  }
  dim.resize(dim.nd+1);
  dim.set(dim.nd-1, max_len);
  // TODO: we don't want to do this for CPU computation
  ids_host = (unsigned*)malloc(dim.bd * dim[dim.nd-1] * sizeof(unsigned));
}

Dim LookupSequenceNode::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

// TODO: This should be made more efficient on GPU, and also remove the dependency
//       on host memory so we don't have to keep the memory around.
void LookupSequenceNode::accumulate_grad(const Tensor& g) {
  Tensor gb(params.get()->dim, g.v, g.device, g.mem_pool);
  size_t one_size = gb.d.size(), num_steps = (dim.bd * dim[dim.nd-1]);
  for(size_t i = 0; i < num_steps; ++i) {
    params.get()->accumulate_grad(ids_host[i], gb);
    gb.v += one_size;
  }
}

LookupSequenceNode::~LookupSequenceNode() {
  free(ids_host);
}

#endif

template<class MyDevice>
void ConstParameterNode::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.tvec().device(*dev.edevice) = params.get()->values.tvec() * params.mp->weight_decay.current_weight_decay();
}

template<class MyDevice>
void ConstParameterNode::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node: i = " << i << endl;
  abort();
}
DYNET_NODE_INST_DEV_IMPL(ConstParameterNode)

template<class MyDevice>
void ParameterNode::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
// TODO
//  if (params->not_regularized) {
//    fx.v = params->values.v;
//    return;
//  }
  fx.tvec().device(*dev.edevice) = params.get()->values.tvec() * params.mp->weight_decay.current_weight_decay();
}

template<class MyDevice>
void ParameterNode::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node: i = " << i << endl;
  abort();
}
DYNET_NODE_INST_DEV_IMPL(ParameterNode)

template<class MyDevice>
void InputNode::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
#if __CUDACC__
  cudaMemcpyAsync(fx.v, &pdata->front(), dim.size() * sizeof(float), cudaMemcpyHostToDevice);
#else
  // TODO memcpy is only necessary if pdata->front() points to an unaligned location
  // need to compute this value
  bool is_input_address_aligned = false;
  if (!is_input_address_aligned) {
    memcpy(fx.v, &pdata->front(), dim.size() * sizeof(float));
  } else {
    fx.v = const_cast<float*>(&pdata->front());
  }
#endif
}

template<class MyDevice>
void InputNode::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}
DYNET_NODE_INST_DEV_IMPL(InputNode)

template<class MyDevice>
void SparseInputNode::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.tvec().device(*dev.edevice) = fx.tvec().constant(defdata);
#if __CUDACC__
  unsigned int* ids_ptr = (unsigned int*)aux_mem;
  float* data_ptr = (float*)(ids_ptr + ids.size());
  cudaMemcpyAsync(ids_ptr, &ids[0], ids.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(data_ptr, &data[0], data.size() * sizeof(float), cudaMemcpyHostToDevice);
  dynet::gpu::sparse_assign(ids.size(), ids_ptr, data_ptr, fx.v);
#else
  for(size_t i = 0; i < ids.size(); ++i)
    fx.v[ids[i]] = data[i];
#endif
}

template<class MyDevice>
void SparseInputNode::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}
DYNET_NODE_INST_DEV_IMPL(SparseInputNode)

template<class MyDevice>
void ScalarInputNode::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
#if __CUDACC__
  cudaMemcpyAsync(fx.v, pdata, 1 * sizeof(float), cudaMemcpyHostToDevice);
#else
  fx.v[0] = *pdata;
#endif
}

template<class MyDevice>
void ScalarInputNode::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}
DYNET_NODE_INST_DEV_IMPL(ScalarInputNode)

template<class MyDevice>
void LookupNode::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  if(pindex) {
    assert(*pindex < params.get()->values.size());
    assert (fx.d.batch_elems() == 1);
    fx.tvec().device(*dev.edevice) = params.get()->values[*pindex].tvec() * params.mp->weight_decay.current_weight_decay();
  } else {
    assert (pindices);
    assert (fx.d.batch_elems() == pindices->size());
#if __CUDACC__
    CUDA_CHECK(cudaMemcpyAsync((unsigned*)aux_mem, &(*pindices)[0], fx.d.bd * sizeof(unsigned), cudaMemcpyHostToDevice));
    dynet::gpu::sparse_lookup(fx.d.bd, (unsigned*)aux_mem, fx.d.batch_size(), params.mp->weight_decay.current_weight_decay(), params.get()->all_values.v, fx.v);
#else
    for (unsigned b = 0; b < pindices->size(); ++b) {
      unsigned i = pindices->at(b);
      assert (i < params.get()->values.size());
      fx.tb<2>().chip<2>(b).device(*dev.edevice) = params.get()->values[i].tb<2>() * params.mp->weight_decay.current_weight_decay();
    }
#endif
  }
}

template<class MyDevice>
void LookupNode::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}
DYNET_NODE_INST_DEV_IMPL(LookupNode)

template<class MyDevice>
void LookupSequenceNode::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  size_t num_words = dim[dim.nd-1], num_steps = dim.bd * num_words, p_size = params.get()->dim.size();
  // Lay out the IDs in memory
  memset(ids_host, 0, num_steps * sizeof(unsigned));
  if(pindex) {
    assert(pindex->size() == num_words);
    memcpy(ids_host, &(*pindex)[0], pindex->size() * sizeof(unsigned));
  } else {
    assert(pindices->size() == dim.bd);
    for(size_t b = 0; b < dim.bd; ++b) {
      assert((*pindices)[b].size() <= num_words);
      memcpy(ids_host + num_words, &((*pindices)[b][0]), (*pindices)[b].size() * sizeof(unsigned));
    }
  }
#if __CUDACC__
  CUDA_CHECK(cudaMemcpyAsync((unsigned*)aux_mem, ids_host, num_steps * sizeof(unsigned), cudaMemcpyHostToDevice));
  dynet::gpu::sparse_lookup(num_steps, (unsigned*)aux_mem, p_size, params.mp->weight_decay.current_weight_decay(), params.get()->all_values.v, fx.v);
#else
  float* p_ptr = params.get()->all_values.v;
  for(size_t i = 0; i < num_steps; ++i) 
    memcpy(fx.v + p_size * i, p_ptr + p_size * ids_host[i], p_size * sizeof(float));
  fx.tvec().device(*dev.edevice) = fx.tvec() * params.mp->weight_decay.current_weight_decay();
#endif
}

template<class MyDevice>
void LookupSequenceNode::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}
DYNET_NODE_INST_DEV_IMPL(LookupSequenceNode)

} // namespace dynet
