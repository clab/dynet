#ifndef CNN_EDGES_H_
#define CNN_EDGES_H_

#include "cnn/cnn.h"

namespace cnn {

// y = x_1 * x_2
struct MatrixMultiply : public Edge {
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
};

// y = x_1 \sum_{i=2, 4 ...} A_i * x_{i+1}
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

struct PickElement : public Edge {
  // x_1 is a vector
  // x_2 is a scalar index stored in (0,0)
  // y = (x_1)_{x_2}
  // this is used to implement cross-entropy training
  string as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "pick(" << arg_names[0] << '_' << arg_names[1] << ')';
    return s.str();
  }

  Matrix forward(const vector<const Matrix*>& xs) const {
    assert(xs.size() == 2);
    const Matrix& x = *xs.front();
    assert(x.cols() == 1);
    const Matrix& mindex = *xs.back();
    assert(mindex.rows() == 1);
    assert(mindex.cols() == 1);
    const unsigned index = static_cast<unsigned>(mindex(0,0));
    assert(index < x.rows());
    Matrix fx(1,1);
    fx(0,0) = x(index, 0);
    return fx;
  }

  // derivative is 0 in all dimensions except 1 for the selected element
  Matrix backward(const vector<const Matrix*>& xs,
                    const Matrix& fx,
                    const Matrix& dEdf,
                    unsigned i) const override {
    assert(i == 0); // f with respect to x_2 is not smooth
    assert(dEdf.rows() == 1);
    assert(dEdf.cols() == 1);
    const Matrix& x = *xs.front();
    const Matrix& mindex = *xs.back();

    // TODO should be sparse
    Matrix dEdx1 = Matrix::Zero(x.rows(), 1); 
    dEdx1(mindex(0,0),0) = dEdf(0,0);
    return dEdx1;
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
