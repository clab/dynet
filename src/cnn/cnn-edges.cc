#include "cnn/cnn-edges.h"

#include <sstream>

using namespace std;

namespace cnn {

bool ParameterEdge::has_parameters() const { return true; }

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

string MatrixMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << arg_names[1];
  return s.str();
}

Matrix MatrixMultiply::forward(const vector<const Matrix*>& xs) const {
  assert(xs.size() == 2);
  return (*xs[0]) * (*xs[1]);
}

Matrix MatrixMultiply::backward(const vector<const Matrix*>& xs,
                                const Matrix& fx,
                                const Matrix& dEdf,
                                unsigned i) const {
  assert(i < 2);
  if (i == 0) {
    return dEdf * xs[1]->transpose();
  } else {
    return xs[0]->transpose() * dEdf;
  }
}


} // namespace cnn
