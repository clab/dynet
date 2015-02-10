#ifndef CNN_PARAM_EDGES_H_
#define CNN_PARAM_EDGES_H_

#include "cnn/cnn.h"
#include "cnn/params.h"

namespace cnn {

struct ParameterEdgeBase : public Edge {
  virtual void accumulate_grad(const Matrix& g) = 0;
};

// represents optimizable parameters
struct ParameterEdge : public ParameterEdgeBase {
  explicit ParameterEdge(Parameters* p) : dim(p->values.rows(), p->values.cols()), params(p) {}
  bool has_parameters() const override;
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  void accumulate_grad(const Matrix& g) override;
  Dim dim;
  Parameters* params;
};

// represents constant inputs
struct InputEdge : public ParameterEdgeBase {
  explicit InputEdge(ConstParameters* p) : dim(p->values.rows(), p->values.cols()), params(p) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  void accumulate_grad(const Matrix& g) override;
  Dim dim;
  ConstParameters* params;
};

// represents a matrix/vector embedding of an item of a discrete set (1-hot coding)
struct LookupEdge : public ParameterEdgeBase {
  LookupEdge(LookupParameters* p) : dim(p->dim), index(), params(p) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  void accumulate_grad(const Matrix& g) override;
  Dim dim;
  unsigned index;
  LookupParameters* params;
};

} // namespace cnn

#endif
