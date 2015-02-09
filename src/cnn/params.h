#ifndef CNN_PARAMS_H_
#define CNN_PARAMS_H_

#include <vector>
#include "cnn/tensor.h"

namespace cnn {

struct ParametersBase {
  virtual ~ParametersBase();
  virtual size_t size() const = 0;
};

// represents a small set of parameters (e.g., a weight matrix)
struct Parameters : public ParametersBase {
  explicit Parameters(const Dim& d) : dim(d), values(Random(d)) {}
  size_t size() const override;
  real& operator()(int i, int j) { return values(i,j); }
  const real& operator()(int i, int j) const { return values(i,j); }

  Dim dim;
  Matrix values;
};

// represents a matrix/vector embedding of a discrete set
struct LookupParameters : public ParametersBase {
  LookupParameters(unsigned n, const Dim& d) : dim(d), index(), values(n) {
    for (auto& v : values) v = Random(d);
  }
  size_t size() const override;
  const Matrix& embedding() const { return values[index]; }
  Matrix& operator[](unsigned i) { return values[i]; }
  const Matrix& operator[](unsigned i) const { return values[i]; }

  Dim dim;
  unsigned index; // index of item in set to be embedded
  std::vector<Matrix> values;
};

} // namespace cnn

#endif
