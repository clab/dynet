#ifndef CNN_EDGES_H_
#define CNN_EDGES_H_

#include "cnn/cnn.h"
#include "cnn/eigen-backend.h"

namespace cnn {

struct Concatenate : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  // src_row_indices[i] says what row in fx the ith x vector was assigned to
  // used to simplify backprop
  mutable std::vector<unsigned> src_row_indices;
};

// Let x be a vector-valued input, x_i represents the score of the ith element, then
// y = \sum{i != element} max{0, margin - x_element + x_i}
struct Hinge : public Edge {
  explicit Hinge(unsigned e, real m = 1.0) : element(e), pelement(&element), margin(m) {}
  explicit Hinge(unsigned* pe, real m = 1.0) : element(), pelement(pe), margin(m) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  unsigned element;
  const unsigned* pelement;
  real margin;
  mutable Matrix u; // partial forward values
};

struct Identity : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
};

// hyperparameter: width > 1
// x_1 is a vector in R^n, which we write x
// y is a vector in R^{n / width}
// y_i = max_{x_{i * width - width + 1}, ..., x_{i * width}}
struct MaxPooling1D : public Edge {
  MaxPooling1D(unsigned w) : width(w) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  unsigned width;
  mutable std::vector<unsigned> ind;
};

// y = x_1 * x_2
struct MatrixMultiply : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
};

// y = x_1 \cdot x_2  (Hadamard product)
struct CwiseMultiply : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
};

// y = x_1 \sum_{i=2, 4 ...} A_i * x_{i+1}
// NOTE: if A_i is a vector then * computes the component-wise product
// this is an ugly hack to deal with diagonal matrices
struct Multilinear : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
};

// y = -x_1
struct Negate : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
};

// y = max(0,x)
struct Rectify : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
};

// y = hardtanh(0,x)
struct HardTanh : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
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
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  real target_y;
  real* ptarget_y;
};

// TODO move implementations of everything here into .cc file see MatrixMultiply as an example
using namespace std;

struct Sum : public Edge {
  // y = \sum_i x_i
  string as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << arg_names[0];
    for (unsigned i = 1; i < tail.size(); ++i)
      s << " + " << arg_names[i];
    return s.str();
  }

  Matrix forward(const vector<const Matrix*>& xs) const {
    assert(xs.size() > 0);
    Matrix res = *xs[0];
    for (unsigned i = 1; i < xs.size(); ++i)
      res += *xs[i];
    return res;
  }
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override {
    return dEdf;
  }
};

struct SquaredEuclideanDistance : public Edge {
  // y = || x_1 - x_2 ||^2
  string as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||^2";
    return s.str();
  }

  Matrix forward(const vector<const Matrix*>& xs) const {
    assert(xs.size() == 2);
    Matrix res(1,1);
    res(0,0) = (*xs[0] - *xs[1]).squaredNorm();
    return res;
  }
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override {
    assert(i < 2);
    real scale = dEdf(0,0) * 2;
    if (i == 1) scale = -scale;
    return scale * (*xs[0] - *xs[1]);
  }
};

struct LogisticSigmoid : public Edge {
  // y = \sigma(x_1)
  string as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "\\sigma(" << arg_names[0] << ')';
    return s.str();
  }

  Matrix forward(const vector<const Matrix*>& xs) const {
    assert(xs.size() == 1);
    const Matrix& x = *xs.front();
    return Elewise::SigmoidForward(x);
  }
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override {
    assert(i == 0);
    const Matrix& x = *xs.front();
    return Elewise::SigmoidBackward(dEdf, fx, x);
  }
};

struct Tanh : public Edge {
  // y = tanh x_1
  string as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "tanh(" << arg_names[0] << ')';
    return s.str();
  }

  Matrix forward(const vector<const Matrix*>& xs) const {
    assert(xs.size() == 1);
    const Matrix& x = *xs.front();
    return Elewise::TanhForward(x);
  }
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override {
    assert(i == 0);
    const Matrix& x = *xs.front();
    return Elewise::TanhBackward(dEdf, fx, x);
  }
};

// z = \sum_j \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct LogSoftmax : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const vector<const Matrix*>& xs) const override;
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override;
};

// z = \sum_{j \in denom} \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct RestrictedLogSoftmax : public Edge {
  explicit RestrictedLogSoftmax(const std::vector<unsigned>& d) : denom(d) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const vector<const Matrix*>& xs) const override;
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
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
  string as_string(const vector<string>& arg_names) const override;
  Matrix forward(const vector<const Matrix*>& xs) const override;
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override;
  unsigned val;
  const unsigned* pval;
};

struct Square : public Edge {
  // y = x_1 \odot x_1
  // assumption: x_1 is a vector
  string as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "square(" << arg_names[0] << ')';
    return s.str();
  }

  Matrix forward(const vector<const Matrix*>& xs) const {
    assert(xs.size() == 1); // just a single input
    const Matrix& x = *xs.front();
    return x.cwiseProduct(x);
  }
  Matrix backward(const vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override {
    assert(i == 0);
    return dEdf.cwiseProduct(*xs.front()) * 2;
  }
};

} // namespace cnn

#endif
