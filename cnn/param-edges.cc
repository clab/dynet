#include "cnn/param-edges.h"

#include <sstream>

using namespace std;

namespace cnn {

bool ParameterEdge::has_parameters() const { return true; }

string ParameterEdge::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "params" << dim;
  return s.str();
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
  s << "inputsN(" << m.rows() << ',' << m.cols() << ')';
  return s.str();
}

Tensor InputEdge::forward(const vector<const Tensor*>& xs) const {
  assert(xs.size() == 0);
  return m;
}

Tensor InputEdge::backward(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const {
  cerr << "called backward() on arity 0 edge\n";
  abort();
}

string LookupEdge::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "lookup[|x|=" << params->values.size() << " --> " << dim << ']';
  return s.str();
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
