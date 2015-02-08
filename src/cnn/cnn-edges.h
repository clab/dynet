#ifndef CNN_EDGES_H_
#define CNN_EDGES_H_

#include "cnn/cnn.h"

namespace cnn {

// represents optimizable parameters
struct ParameterEdge : public Edge {
  ParameterEdge(const Dim& d) : dim(d), values(Random(d)) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  inline real& operator()(int i, int j) { return values(i,j); }
  inline const real& operator()(int i, int j) const { return values(i,j); }
  Dim dim;
  Matrix values;
};

// represents constant inputs
struct InputEdge : public Edge {
  InputEdge(const Dim& d) : dim(d), values(Zero(d)) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  inline real& operator()(int i, int j) { return values(i,j); }
  inline const real& operator()(int i, int j) const { return values(i,j); }
  Dim dim;
  Matrix values;
};

using namespace std; // TODO get rid of this, move implementations of virtual functions into .cc file

struct MatrixMultiply : public Edge {
  // y = x_1 * x_2
  std::string as_string(const std::vector<std::string>& arg_names) const {
    ostringstream s;
    s << arg_names[0] << " * " << arg_names[1];
    return s.str();
  }

  Matrix forward(const std::vector<const Matrix*>& xs) const {
    assert(xs.size() == 2);
    return (*xs[0]) * (*xs[1]);
  }
  Matrix backward(const std::vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override {
    assert(i < 2);
    if (i == 0) {
      return dEdf * xs[1]->transpose();
    } else {
      return xs[0]->transpose() * dEdf;
    }
  }
};

struct Sum : public Edge {
  // y = \sum_i x_i
  string as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << arg_names[0];
    for (unsigned i = 1; i < tail.size(); ++i)
      s << " + " << arg_names[1];
    return s.str();
  }

  Matrix forward(const vector<const Matrix*>& xs) const {
    assert(xs.size() > 1);
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

struct EuclideanDistance : public Edge {
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
  // y = tanh x_1
  string as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "\\sigma(" << arg_names[0] << ')';
    return s.str();
  }

  Matrix forward(const vector<const Matrix*>& xs) const {
    assert(xs.size() == 1);
    const Matrix& x = *xs.front();
    const unsigned rows = x.rows();
    const unsigned cols = x.cols();
    Matrix fx(rows, cols);
    for (unsigned i = 0; i < rows; ++i)
      for (unsigned j = 0; j < cols; ++j)
        fx(i,j) = 1. / (1. + exp(-x(i,j)));
    return fx;
  }
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override {
    assert(i == 0);
    const Matrix& x = *xs.front();
    const unsigned rows = x.rows();
    const unsigned cols = x.cols();
    Matrix dfdx(rows, cols);
    for (unsigned i = 0; i < rows; ++i)
      for (unsigned j = 0; j < cols; ++j)
        dfdx(i,j) = (1. - fx(i,j)) * fx(i,j);
    return dfdx.cwiseProduct(dEdf);
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
    const unsigned rows = x.rows();
    const unsigned cols = x.cols();
    Matrix fx(rows, cols);
    for (unsigned i = 0; i < rows; ++i)
      for (unsigned j = 0; j < cols; ++j)
        fx(i,j) = tanh(x(i,j));
    return fx;
  }
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override {
    assert(i == 0);
    const Matrix& x = *xs.front();
    const unsigned rows = x.rows();
    const unsigned cols = x.cols();
    Matrix dfdx(rows, cols);
    for (unsigned i = 0; i < rows; ++i)
      for (unsigned j = 0; j < cols; ++j)
        dfdx(i,j) = 1. - fx(i,j) * fx(i,j);
    return dfdx.cwiseProduct(dEdf);
  }
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
