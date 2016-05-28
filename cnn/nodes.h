#ifndef CNN_NODES_H_
#define CNN_NODES_H_

#include "cnn/cnn.h"

namespace cnn {

// y = L_sparsemax(x_0; q)
// where x_0 is a vector of "unnormalized" probabilities
// q are the vector of labels
struct SparsemaxLoss : public Node {
  explicit SparsemaxLoss(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& target) : Node(a), q(target), pq(&q) {}
  explicit SparsemaxLoss(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* ptarget) : Node(a), q(), pq(ptarget) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const override;
  const std::vector<unsigned> q;
  const std::vector<unsigned>* pq;
};

// y = sparsemax(x)
// y = arg min_y ||y - x||^2
struct Sparsemax : public Node {
  explicit Sparsemax(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const override;
};

// M = x_0, v = x_1
// y = M + v (broadcasting over columns)
struct AddMv : public Node {
  explicit AddMv(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const override;
};

// y = inv(x)
// x = an invertible matrix
struct MatrixInverse : public Node {
  explicit MatrixInverse(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const override;
};

// y = select_rows(x, rows)
// x = a matrix
struct SelectRows : public Node {
  explicit SelectRows(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& r) : Node(a), rows(r), prows(&rows) {}
  explicit SelectRows(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pr) : Node(a), prows(pr) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const override;
  std::vector<unsigned> rows;
  const std::vector<unsigned>* prows;
};

// y = select_cols(x, cols)
// x = a matrix
struct SelectCols : public Node {
  explicit SelectCols(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& c) : Node(a), cols(c), pcols(&cols) {}
  explicit SelectCols(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pc) : Node(a), pcols(pc) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const override;
  std::vector<unsigned> cols;
  const std::vector<unsigned>* pcols;
};

// y = pow(x_1, x_2)
// x_2 raise every element in x_1 to the power of scalar x_2
struct Pow : public Node {
  explicit Pow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const override;
};

// y = min{x_1, x_2}
struct Min : public Node {
  explicit Min(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

// y = max{x_1, x_2}
struct Max : public Node {
  template <typename T> explicit Max(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

// y = Tr(x_1 * x_2^T)
struct TraceOfProduct : public Node {
  explicit TraceOfProduct(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

// y = alpha * x_1
struct ConstScalarMultiply : public Node {
  explicit ConstScalarMultiply(const std::initializer_list<VariableIndex>& a, float alpha) : Node(a), alpha(alpha) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
  float alpha;
};

// y = x_1^T . x_2
struct DotProduct : public Node {
  explicit DotProduct(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

// y = x_1^T
// NOTE: if you have a column or row vector as input, runtime is constant
// if you have a matrix as input, the runtime is O(mn) - try to avoid using this
struct Transpose : public Node {
  explicit Transpose(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = reshape(x_1, --> to)
struct Reshape : public Node {
  explicit Reshape(const std::initializer_list<VariableIndex>& a, const Dim& to) : Node(a), to(to) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  Dim to;
};

// with a single argument x \in R^{n x m}
// y_i = \sum_j x_i,j
// if you want to reweight the columns and then sum them, use MatrixMultiply
struct SumColumns : public Node {
  template <typename T> explicit SumColumns(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

// y_i = \sum_{j=1}^n x_1:{i-1+j}
struct KMHNGram : public Node {
  explicit KMHNGram(const std::initializer_list<VariableIndex>& a, unsigned n) : Node(a), n(n) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  unsigned n;  // width, n=2 for Karl's paper
};

// Forward:
//   Y_ij = A_ijk * B_k + C_ij
//
// Backward:
//   (dE/dA)_ijk = (dE/dY)_ij * L_k
//   (dE/dB)_k = (dE/dY)_ij * A_ijk
//   (dE/dC)_ij = (dE/dY)_ij
struct InnerProduct3D_1D : public Node {
  InnerProduct3D_1D(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

//   Y_i = A_ijk * B_k * C_j
struct InnerProduct3D_1D_1D : public Node {
  InnerProduct3D_1D_1D(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const override;
};

// n_{i,j} ~ N(0,stddev)
// y = x + n
struct GaussianNoise : public Node {
  explicit GaussianNoise(const std::initializer_list<VariableIndex>& a, real stddev) : Node(a), stddev(stddev) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  real stddev;
};

// y = dropout(x,p) where p specifies the dropout probability
struct Dropout : public Node {
  explicit Dropout(const std::initializer_list<VariableIndex>& a, real p) : Node(a), p(p) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
  real p;
};

// y = block_dropout(x,p) where p specifies the probability for dropping-out the entire block
struct BlockDropout : public Node {
  explicit BlockDropout(const std::initializer_list<VariableIndex>& a, real p) : Node(a), dropout_probability(p) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
  real dropout_probability;
};

// y = c + x_1
// (c is a vector or matrix of the constant, usually 1, but can be configured)
struct ConstantPlusX : public Node {
  explicit ConstantPlusX(const std::initializer_list<VariableIndex>& a, real o) : Node(a), c(o) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  real c;
};

// y = c - x_1
// (c is a vector or matrix of the constant, usually 1, but can be configured)
struct ConstantMinusX : public Node {
  explicit ConstantMinusX(const std::initializer_list<VariableIndex>& a, real o) : Node(a), c(o) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  real c;
};

// y = sqrt x_1
struct Sqrt : public Node {
  explicit Sqrt(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = erf x_1
struct Erf : public Node {
  explicit Erf(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = tanh x_1
struct Tanh : public Node {
  explicit Tanh(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = x_1 \odot x_1
struct Square : public Node {
  explicit Square(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = x_1 \odot x_1 \odot x_1
struct Cube : public Node {
  explicit Cube(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = exp x_1
struct Exp : public Node {
  explicit Exp(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = lgamma x_1
struct LogGamma : public Node {
  explicit LogGamma(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const override;
};

// y = log x_1  (base e, i.e., natural log)
struct Log : public Node {
  explicit Log(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// concatenate rows
struct Concatenate : public Node {
  template <typename T> explicit Concatenate(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  // src_row_indices[i] says what row in fx the ith x vector was assigned to
  // used to simplify backprop
  mutable std::vector<unsigned> src_row_indices;
};

// concatenate column vectors into a matrix
// x_i must be a column vector in R^n
struct ConcatenateColumns : public Node {
  template <typename T> explicit ConcatenateColumns(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// x_1 is a scalar (or row vector)
// x_2 is a scalar (or row vector)
// y = max(0, margin - x_1 + x_2)
struct PairwiseRankLoss : public Node {
  explicit PairwiseRankLoss(const std::initializer_list<VariableIndex>& a, real m = 1.0) : Node(a), margin(m) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  real margin;
};

// Let x be a vector-valued input, x_i represents the score of the ith element, then
// y = \sum{i != element} max{0, margin - x_element + x_i}
struct Hinge : public Node {
  explicit Hinge(const std::initializer_list<VariableIndex>& a, unsigned e, real m = 1.0) : Node(a), element(e), pelement(&element), margin(m) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const unsigned* pe, real m = 1.0) : Node(a), element(), pelement(pe), margin(m) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  unsigned element;
  const unsigned* pelement;
  real margin;
};

// y = x_1, but dy/dx is set to 0
struct NoBackprop : public Node {
  explicit NoBackprop(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = x_1
struct Identity : public Node {
  explicit Identity(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// hyperparameter: width > 1
// x_1 is a vector in R^n, which we write x
// y is a vector in R^{n / width}
// y_i = max_{x_{i * width - width + 1}, ..., x_{i * width}}
struct MaxPooling1D : public Node {
  MaxPooling1D(const std::initializer_list<VariableIndex>& a, unsigned w) : Node(a), width(w) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  unsigned width;
  mutable std::vector<unsigned> ind;
};

// y = x_1 * x_2
struct MatrixMultiply : public Node {
  explicit MatrixMultiply(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = x_1 \cdot x_2  (Hadamard product)
struct CwiseMultiply : public Node {
  explicit CwiseMultiply(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = x_1 / x_2  (cwiseQuotient)
struct CwiseQuotient : public Node {
  explicit CwiseQuotient(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

// y = x_1 \sum_{i=2, 4 ...} A_i * x_{i+1}
struct AffineTransform : public Node {
  template <typename T> explicit AffineTransform(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = -x_1
struct Negate : public Node {
  explicit Negate(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = max(0,x)
struct Rectify : public Node {
  explicit Rectify(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// you could do this with LogisticSigmoid, Softmax or a variety of other
// functions, but this is often useful.
// x_1 must be a vector with values between 0 and 1
// target_y is an equivalently sized vector w values between 0 and 1
// y = ty * log(x_1) + (1 - ty) * log(x_1)
struct BinaryLogLoss : public Node {
  BinaryLogLoss(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = \log \sum_i \exp x_i
// done in log space carefully to avoid over/underflow issues
struct LogSumExp : public Node {
  template <typename T> explicit LogSumExp(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
};

struct LogDet : public Node {
  template <typename T> explicit LogDet(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  //size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
};

// y = \sum_i x_i
struct Sum : public Node {
  template <typename T> explicit Sum(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  // TODO: Sum should be be implemented over the entire mini-batch, but this is not
  //       super-easy in the current implementation
  // virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
};

// y = \sum_i x_i
struct SumBatches : public Node {
  template <typename T> explicit SumBatches(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
};

// y = ( \sum_i x_i ) / |x|
struct Average : public Node {
  template <typename T> explicit Average(const T& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
};

// this is used to implement poisson regression
// x_1 = log predicted mean
// ty = true y (this is not a VariableIndex since it has to be a nonnegative integer and
//              is therefore nondifferentiable. There are various continuous extensions
//              using the incomplete gamma function that could be used, but meh)
// y = log Poisson(ty; \lambda = \exp x_1)
//   = ty*x_1 - exp(x_1) - log(ty!)
struct PoissonRegressionLoss : public Node {
  explicit PoissonRegressionLoss(const std::initializer_list<VariableIndex>& a, unsigned true_y) : Node(a), ty(true_y), pty(&ty) {}
  explicit PoissonRegressionLoss(const std::initializer_list<VariableIndex>& a, const unsigned* ptrue_y) : Node(a), ty(), pty(ptrue_y) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
 private:
  unsigned ty;
  const unsigned* pty;
};

// y = || x_1 ||^2
struct SquaredNorm : public Node {
  explicit SquaredNorm(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = || x_1 - x_2 ||^2
struct SquaredEuclideanDistance : public Node {
  explicit SquaredEuclideanDistance(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = || x_1 - x_2 ||_H(d)
struct HuberDistance : public Node {
  explicit HuberDistance(const std::initializer_list<VariableIndex>& a, float d = 1.345f) : Node(a), d(d) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
  float d;
};

// y = || x_1 - x_2 ||_1
struct L1Distance : public Node {
  explicit L1Distance(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
};

// y = \sigma(x_1)
struct LogisticSigmoid : public Node {
  explicit LogisticSigmoid(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
};

// y = x / (1 + |x|)
struct SoftSign : public Node {
  explicit SoftSign(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
};

// z = \sum_j \exp (x_i)_j
// y_i = (x_1)_i / z
struct Softmax : public Node {
  explicit Softmax(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
};

// z = \sum_j \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct LogSoftmax : public Node {
  explicit LogSoftmax(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
};

// z = \sum_j \exp (x_i)_j
// y = (x_1)_element - \log z
struct PickNegLogSoftmax : public Node {
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, unsigned v) : Node(a), val(v), pval(&val), vals(), pvals() {}
  // use this constructor if you want to perform mini-batching
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& v) : Node(a), val(), pval(), vals(v), pvals(&vals) {}
  // use these constructors if you want to change the value after the graph is constructed
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, const unsigned* pv) : Node(a), val(), pval(pv), vals(), pvals() {}
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pv) : Node(a), val(), pval(), vals(), pvals(pv) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
  mutable float* logz;
  unsigned val;
  const unsigned* pval;
  std::vector<unsigned> vals;
  const std::vector<unsigned>* pvals;
};

// z = \sum_{j \in denom} \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct RestrictedLogSoftmax : public Node {
  explicit RestrictedLogSoftmax(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& d) : Node(a), denom(d) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
  std::vector<unsigned> denom;
};

// x_1 is a vector
// y = (x_1)_{*pval}
// this is used to implement cross-entropy training
struct PickElement : public Node {
  explicit PickElement(const std::initializer_list<VariableIndex>& a, unsigned v) : Node(a), val(v), pval(&val), vals(), pvals() {}
  // use this constructor if you want to perform mini-batching
  explicit PickElement(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& v) : Node(a), val(), pval(), vals(v), pvals(&vals) {}
  // use these constructors if you want to change the value after the graph is constructed
  explicit PickElement(const std::initializer_list<VariableIndex>& a, const unsigned* pv) : Node(a), val(), pval(pv), vals(), pvals() {}
  explicit PickElement(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pv) : Node(a), val(), pval(), vals(), pvals(pv) {}
  virtual bool supports_multibatch() const override { return true; }
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
  unsigned val;
  const unsigned* pval;
  std::vector<unsigned> vals;
  const std::vector<unsigned>* pvals;
};

// x_1 is a vector
// y = x_1[start:end]
// (start inclusive, end exclusive)
struct PickRange : public Node {
  explicit PickRange(const std::initializer_list<VariableIndex>& a, unsigned start, unsigned end) : Node(a), start(start), end(end) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const override;
  unsigned start;
  unsigned end;
};

// represents a simple vector of 0s
struct Zeroes : public Node {
  explicit Zeroes(const Dim& d) : dim(d) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  Dim dim;
};

} // namespace cnn

#endif
