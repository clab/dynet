#ifndef CNN_PARAM_EDGES_H_
#define CNN_PARAM_EDGES_H_

#include "cnn/cnn.h"
#include "cnn/model.h"

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

// represents specified (not learned) inputs to the network
struct InputEdge : public Edge {
  explicit InputEdge(const Dim& d) : m(d.rows, d.cols) {}
  explicit InputEdge(const Matrix& mm) : m(mm) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  Matrix m;
};

// represents a matrix/vector embedding of an item of a discrete set (1-hot coding)
struct LookupEdge : public ParameterEdgeBase {
  LookupEdge(LookupParameters* p, unsigned ind) : dim(p->dim), index(ind), pindex(&index), params(p), has_optimizable_parameters(true) {}
  LookupEdge(LookupParameters* p, unsigned* pind) : dim(p->dim), index(), pindex(pind), params(p), has_optimizable_parameters(true) {}
  bool has_parameters() const override;
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Matrix forward(const std::vector<const Matrix*>& xs) const override;
  Matrix backward(const std::vector<const Matrix*>& xs,
                  const Matrix& fx,
                  const Matrix& dEdf,
                  unsigned i) const override;
  void accumulate_grad(const Matrix& g) override;
  Dim dim;
  unsigned index;
  unsigned* pindex;
  LookupParameters* params;
  bool has_optimizable_parameters;
};

} // namespace cnn

#endif
