#include "cnn/param-nodes.h"
#include "cnn/tensor.h"

#include <sstream>

using namespace std;

namespace cnn {

string ParameterNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "parameters(" << dim << ", " << params << ')';
  return s.str();
}

Dim ParameterNode::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 0);
  return dim;
}

void ParameterNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.v = params->values.v;
}

void ParameterNode::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node: i = " << i << endl;
  abort();
}

void ParameterNode::accumulate_grad(const Tensor& g) {
  params->accumulate_grad(g);
}

string InputNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "constant(" << dim << ')';
  return s.str();
}

Dim InputNode::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

void InputNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
#if HAVE_CUDA
  cudaMemcpyAsync(fx.v, &pdata->front(), dim.size() * sizeof(float), cudaMemcpyHostToDevice);
#else
  memcpy(fx.v, &pdata->front(), dim.size() * sizeof(float));
#endif
}

void InputNode::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}

string ScalarInputNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "scalar_constant(" << pdata << ')';
  return s.str();
}

Dim ScalarInputNode::dim_forward(const vector<Dim>& xs) const {
  return Dim({1});
}

void ScalarInputNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
#if HAVE_CUDA
  cudaMemcpyAsync(fx.v, pdata, 1 * sizeof(float), cudaMemcpyHostToDevice);
#else
  fx.v[0] = *pdata;
#endif
}

void ScalarInputNode::backward_impl(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}

string LookupNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "lookup_parameters(|x|=" << params->values.size() << " --> " << dim << ')';
  return s.str();
}

Dim LookupNode::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

void LookupNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  assert(*pindex < params->values.size());
  fx.v = params->values[*pindex].v;
}

void LookupNode::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}

void LookupNode::accumulate_grad(const Tensor& g) {
  params->accumulate_grad(*pindex, g);
}

string BatchLookupNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "batch_lookup_parameters(|x|=" << params->values.size() << " --> " << dim << ')';
  return s.str();
}

Dim BatchLookupNode::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

void BatchLookupNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  assert (fx.d.batch_elems() == pindices->size());
  for (unsigned b = 0; b < pindices->size(); ++b) {
    unsigned i = pindices->at(b);
    assert (i < params->values.size());
    float* v = fx.v + fx.d.batch_size() * (b % fx.d.batch_elems());
    memcpy(v, params->values[i].v, fx.d.batch_size() * sizeof(float));
  }
}

void BatchLookupNode::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}

void BatchLookupNode::accumulate_grad(const Tensor& g) {
  const vector<Tensor>& gb = g.batch_elems();
  for (unsigned b = 0; b < pindices->size(); ++b) {
    unsigned i = pindices->at(b);
    assert (i < params->values.size());
    params->accumulate_grad(i, gb[b]);
  }
}

} // namespace cnn
