#include "cnn/param-edges.h"
#include "cnn/tensor.h"

#include <sstream>

using namespace std;

namespace cnn {

bool ParameterEdge::has_parameters() const { return true; }

string ParameterEdge::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "parameters(" << dim << ')';
  return s.str();
}

Dim ParameterEdge::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 0);
  return dim;
}

void ParameterEdge::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.v = params->values.v;
}

void ParameterEdge::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 edge\n";
  abort();
}

void ParameterEdge::accumulate_grad(const Tensor& g) {
  params->accumulate_grad(g);
}

string InputEdge::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "constant(" << dim << ')';
  return s.str();
}

Dim InputEdge::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

void InputEdge::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  assert((int)dim.size() == (int)pdata->size());
  memcpy(fx.v, &pdata->front(), dim.size() * sizeof(float));
}

void InputEdge::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 edge\n";
  abort();
}

string ScalarInputEdge::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "scalar_constant(" << *pdata << ')';
  return s.str();
}

Dim ScalarInputEdge::dim_forward(const vector<Dim>& xs) const {
  return Dim({1});
}

void ScalarInputEdge::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.v[0] = *pdata;
}

void ScalarInputEdge::backward(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 edge\n";
  abort();
}

string LookupEdge::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "lookup_parameters(|x|=" << params->values.size() << " --> " << dim << ')';
  return s.str();
}

Dim LookupEdge::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

void LookupEdge::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  *fx = *params->values[*pindex];
}

void LookupEdge::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 edge\n";
  abort();
}

bool LookupEdge::has_parameters() const {
  return has_optimizable_parameters;
}

void LookupEdge::accumulate_grad(const Tensor& g) {
  assert(has_optimizable_parameters);
  params->accumulate_grad(*pindex, g);
}

} // namespace cnn
