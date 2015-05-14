#include "cnn/param-nodes.h"
#include "cnn/tensor.h"

#include <sstream>

using namespace std;

namespace cnn {

bool ParameterNode::has_parameters() const { return true; }

string ParameterNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "parameters(" << dim << ')';
  return s.str();
}

Dim ParameterNode::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 0);
  return dim;
}

void ParameterNode::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.v = params->values.v;
}

void ParameterNode::backward(const vector<const Tensor*>& xs,
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

void InputNode::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  assert((int)dim.size() == (int)pdata->size());
  memcpy(fx.v, &pdata->front(), dim.size() * sizeof(float));
}

void InputNode::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 edge\n";
  abort();
}

string ScalarInputNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "scalar_constant(" << *pdata << ')';
  return s.str();
}

Dim ScalarInputNode::dim_forward(const vector<Dim>& xs) const {
  return Dim({1});
}

void ScalarInputNode::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.v[0] = *pdata;
}

void ScalarInputNode::backward(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 edge\n";
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

void LookupNode::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  *fx = *params->values[*pindex];
}

void LookupNode::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 edge\n";
  abort();
}

bool LookupNode::has_parameters() const {
  return has_optimizable_parameters;
}

void LookupNode::accumulate_grad(const Tensor& g) {
  assert(has_optimizable_parameters);
  params->accumulate_grad(*pindex, g);
}

} // namespace cnn
