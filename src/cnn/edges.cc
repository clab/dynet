#include "cnn/edges.h"

#include <sstream>

using namespace std;

namespace cnn {

// TODO move the implementation of all the standard functional edges into here

inline real logsumexp(const Matrix& x) {
  real m = x(0,0);
  for (unsigned i = 1; i < x.rows(); ++i) {
    real r = x(i,0);
    if (r > m) m = r;
  }
  real z = 0;
  for (unsigned i = 0; i < x.rows(); ++i)
    z += exp(x(i,0) - m);
  return m + log(z);
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
