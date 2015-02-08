#include "cnn/cnn-edges.h"

#include <sstream>

using namespace std;

namespace cnn {

string ParameterEdge::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "params" << dim;
  return s.str();
}

Matrix ParameterEdge::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 0);
  return values;
}

Matrix ParameterEdge::backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const {
  return Matrix();
}

string InputEdge::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "inputs" << dim;
  return s.str();
}

Matrix InputEdge::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 0);
  return values;
}

Matrix InputEdge::backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const {
  return Matrix();
}

} // namespace cnn
