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
  friend class Model;
  virtual real g_squared_l2norm() const = 0;
  virtual size_t size() const = 0;
  virtual ~ParametersBase();
};

// represents parameters (e.g., a weight matrix)
struct Parameters : public ParametersBase {
  friend class Model;
  real g_squared_l2norm() const override;
  size_t size() const override;

  real& operator()(int i, int j) { return values(i,j); }
  const real& operator()(int i, int j) const { return values(i,j); }

  void accumulate_grad(const Matrix& g);
  void clear();

  Dim dim;
  Matrix values;
  Matrix g;
 private:
  explicit Parameters(const Dim& d) : dim(d), values(Random(d)), g(Zero(d)) {}
  explicit Parameters(const Matrix& v) : dim(v.rows(), v.cols()), values(v), g(Zero(dim)) {}
};

// represents a matrix/vector embedding of a discrete set
struct LookupParameters : public ParametersBase {
  friend class Model;
  real g_squared_l2norm() const override;
  size_t size() const override;

  Matrix& operator[](unsigned i) { return values[i]; }
  const Matrix& operator[](unsigned i) const { return values[i]; }

  void accumulate_grad(unsigned index, const Matrix& g);
  void clear();

  Dim dim;
  std::vector<Matrix> values;
  std::unordered_map<unsigned, Matrix> g;
 private:
  LookupParameters(unsigned n, const Dim& d) : dim(d), values(n) {
    for (auto& v : values) v = Random(d);
  }
};

// this is a collection of parameters
// if you need a matrix of parameters, or a lookup table - ask an instance of this class
// this knows how to serialize itself
// parameters know how to track their gradients, but any extra information (like velocity) will live here
class Model {
 public:
  ~Model();
  Parameters* add_parameters(const Dim& d);  // initialized randomly
  Parameters* add_parameters(const Matrix& m);  // initial value is m
  LookupParameters* add_lookup_parameters(unsigned n, const Dim& d);

  const std::vector<Parameters*>& parameters_list() const { return params; }
  const std::vector<LookupParameters*>& lookup_parameters_list() const { return lookup_params; }

 private:
  std::vector<ParametersBase*> all_params;
  std::vector<Parameters*> params;
  std::vector<LookupParameters*> lookup_params;
};

} // namespace cnn

#endif
