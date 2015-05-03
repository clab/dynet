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
  return dim;
}

Tensor ParameterEdge::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 0);
  return params->values;
}

Tensor ParameterEdge::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const {
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

Tensor InputEdge::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 0);
  assert((int)dim.size() == (int)pdata->size());
  return FromRawData(dim, &pdata->front());
}

Tensor InputEdge::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const {
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

Tensor ScalarInputEdge::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 0);
  return FromRawData(Dim({1}), pdata);
}

Tensor ScalarInputEdge::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const {
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

Tensor LookupEdge::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 0);
  return params->values[*pindex];
}

Tensor LookupEdge::backward(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i) const {
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
