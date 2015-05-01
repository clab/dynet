#ifndef CNN_EDGES_H_
#define CNN_EDGES_H_

#include "cnn/cnn.h"

namespace cnn {

// y = reshape(x_1, from --> to)
struct Reshape : public Edge {
  explicit Reshape(const Dim& from, const Dim& to) : from(from), to(to) {
    assert(from.size() == to.size());
  }
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  Dim from;
  Dim to;
};

// y_i = \sum_{j} x_i,j
struct SumColumns : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// y_i = \sum_{j=1}^n x_1:{i-1+j}
struct KMHNGram : public Edge {
  explicit KMHNGram(unsigned n) : n(n) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  unsigned n;  // width, n=2 for Karl's paper
};

// Forward:
//   Y_ij = A_ijk * B_k + C_ij
//
// Backward:
//   (dE/dA)_ijk = (dE/dY)_ij * L_k
//   (dE/dB)_k = (dE/dY)_ij * A_ijk
//   (dE/dC)_ij = (dE/dY)_ij
struct InnerProduct3D_1D : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// n_{i,j} ~ N(0,stddev)
// y = x + n
struct GaussianNoise : public Edge {
  explicit GaussianNoise(real stddev) : stddev(stddev) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  real stddev;
};

// y = dropout(x,p) where p specifies the dropout probability
struct Dropout : public Edge {
  explicit Dropout(real p) : p(p) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  mutable Tensor noise_mask;
  real p;
};

// y = 1 - x_1
struct OneMinusX : public Edge {
  OneMinusX() : o(1) {}
  explicit OneMinusX(real o) : o(o) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  real o;
};

// y = tanh x_1
struct Tanh : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// y = x_1 \odot x_1
struct Square : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// y = exp x_1
struct Exp : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// y = log x_1  (base e, i.e., natural log)
struct Log : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// concatenate rows
struct Concatenate : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  // src_row_indices[i] says what row in fx the ith x vector was assigned to
  // used to simplify backprop
  mutable std::vector<unsigned> src_row_indices;
};

// concatenate column vectors into a matrix
// x_i must be a column vector in R^n
struct ConcatenateColumns : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// Let x be a vector-valued input, x_i represents the score of the ith element, then
// y = \sum{i != element} max{0, margin - x_element + x_i}
struct Hinge : public Edge {
  explicit Hinge(unsigned e, real m = 1.0) : element(e), pelement(&element), margin(m) {}
  explicit Hinge(unsigned* pe, real m = 1.0) : element(), pelement(pe), margin(m) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  unsigned element;
  const unsigned* pelement;
  real margin;
  mutable Tensor u; // partial forward values
};

// y = x_1
struct Identity : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// hyperparameter: width > 1
// x_1 is a vector in R^n, which we write x
// y is a vector in R^{n / width}
// y_i = max_{x_{i * width - width + 1}, ..., x_{i * width}}
struct MaxPooling1D : public Edge {
  MaxPooling1D(unsigned w) : width(w) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  unsigned width;
  mutable std::vector<unsigned> ind;
};

// y = x_1 * x_2
struct MatrixMultiply : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// y = x_1 \cdot x_2  (Hadamard product)
struct CwiseMultiply : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// y = x_1 \sum_{i=2, 4 ...} A_i * x_{i+1}
// NOTE: if A_i is a vector then * computes the component-wise product
// this is an ugly hack to deal with diagonal matrices
struct Multilinear : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// y = -x_1
struct Negate : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// y = max(0,x)
struct Rectify : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// you could do this with LogisticSigmoid, Softmax or a variety of other
// functions, but this is often useful.
// x_1 must be a scalar that is a value between 0 and 1
// target_y is a value between 0 and 1
// y = ty * log(x_1) + (1 - ty) * log(x_1)
struct BinaryLogLoss : public Edge {
  BinaryLogLoss(real ty) : target_y(ty), ptarget_y(&target_y) {}
  BinaryLogLoss(real* pty) : target_y(), ptarget_y(pty) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  real target_y;
  real* ptarget_y;
};

// y = \sum_i x_i
struct Sum : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const override;
};

// y = || x_1 - x_2 ||^2
struct SquaredEuclideanDistance : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
};

// y = \sigma(x_1)
struct LogisticSigmoid : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const;
  Tensor backward(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const override;
};

// z = \sum_j \exp (x_i)_j
// y_i = (x_1)_i / z
struct Softmax : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const override;
};

// z = \sum_j \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct LogSoftmax : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const override;
};

// z = \sum_{j \in denom} \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct RestrictedLogSoftmax : public Edge {
  explicit RestrictedLogSoftmax(const std::vector<unsigned>& d) : denom(d) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const override;
  std::vector<unsigned> denom;
};

// x_1 is a vector
// y = (x_1)_{*pval}
// this is used to implement cross-entropy training
struct PickElement : public Edge {
  explicit PickElement(unsigned v) : val(v), pval(&val) {}
  // use this constructor if you want to change the value after the graph is constructed
  explicit PickElement(const unsigned* pv) : val(), pval(pv) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const override;
  unsigned val;
  const unsigned* pval;
};

// x_1 is a vector
// y = x_1[start:end]
// (start inclusive, end exclusive)
struct PickRange : public Edge {
  explicit PickRange(unsigned start, unsigned end) : start(start), end(end) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i) const override;
  unsigned start;
  unsigned end;
};

} // namespace cnn

#endif
