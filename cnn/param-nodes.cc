#include "cnn/param-nodes.h"
#include "cnn/tensor.h"

#include <sstream>

using namespace std;

namespace cnn {

string ConstParameterNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "const_parameters(" << dim << ", " << params << ')';
  return s.str();
}

Dim ConstParameterNode::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 0);
  return dim;
}

void ConstParameterNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.v = params->values.v;
}

void ConstParameterNode::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node: i = " << i << endl;
  abort();
}

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
  if(pindex) {
    assert(*pindex < params->values.size());
    assert (fx.d.batch_elems() == 1);
    fx.v = params->values[*pindex].v;
  } else {
    assert (pindices);
    assert (fx.d.batch_elems() == pindices->size());
    for (unsigned b = 0; b < pindices->size(); ++b) {
      unsigned i = pindices->at(b);
      assert (i < params->values.size());
      float* v = fx.v + fx.d.batch_size() * (b % fx.d.batch_elems());
#if HAVE_CUDA
      cudaMemcpyAsync(v, params->values[i].v, fx.d.batch_size() * sizeof(float), cudaMemcpyDeviceToDevice);
#else
      memcpy(v, params->values[i].v, fx.d.batch_size() * sizeof(float));
#endif
    }
  }
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
  if(pindex) {
    params->accumulate_grad(*pindex, g);
  } else {
    assert (pindices);
    const vector<Tensor>& gb = g.batch_elems();
    for (unsigned b = 0; b < pindices->size(); ++b) {
      unsigned i = pindices->at(b);
      assert (i < params->values.size());
      params->accumulate_grad(i, gb[b]);
    }
  }
}

} // namespace cnn
