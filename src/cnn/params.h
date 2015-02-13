#ifndef CNN_PARAMS_H_
#define CNN_PARAMS_H_

#include <vector>
#include <unordered_map>
#include "cnn/tensor.h"

namespace cnn {

// to deal with sparse updates, there are two parameter classes:
// * Parameters represents a vector, matrix, (eventually higher order tensors)
//   of parameters. These are densely updated.
// * LookupParameters represents a table of vectors that are used to embed a
//   set of discrete objects. These are sparsely updated.

struct ParametersBase {
  virtual ~ParametersBase();
  virtual real g_squared_l2norm() const = 0;
  virtual size_t size() const = 0;
};

// represents parameters (e.g., a weight matrix)
struct Parameters : public ParametersBase {
  explicit Parameters(const Dim& d) : dim(d), values(Random(d)), g(Zero(d)) {}
  explicit Parameters(const Matrix& v) : dim(v.rows(), v.cols()), values(v), g(Zero(dim)) {}
  real g_squared_l2norm() const override;
  size_t size() const override;

  real& operator()(int i, int j) { return values(i,j); }
  const real& operator()(int i, int j) const { return values(i,j); }

  void accumulate_grad(const Matrix& g);
  void clear();

  Dim dim;
  Matrix values;
  Matrix g;
};

// represents a matrix/vector embedding of a discrete set
struct LookupParameters : public ParametersBase {
  LookupParameters(unsigned n, const Dim& d) : dim(d), values(n) {
    for (auto& v : values) v = Random(d);
  }
  real g_squared_l2norm() const override;
  size_t size() const override;

  Matrix& operator[](unsigned i) { return values[i]; }
  const Matrix& operator[](unsigned i) const { return values[i]; }

  void accumulate_grad(unsigned index, const Matrix& g);
  void clear();

  Dim dim;
  std::vector<Matrix> values;
  std::unordered_map<unsigned, Matrix> g;
};

} // namespace cnn

#endif
