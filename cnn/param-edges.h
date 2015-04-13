#ifndef CNN_PARAM_EDGES_H_
#define CNN_PARAM_EDGES_H_

#include "cnn/cnn.h"
#include "cnn/model.h"

namespace cnn {

struct ParameterEdgeBase : public Edge {
  virtual void accumulate_grad(const Tensor& g) = 0;
};

// represents optimizable parameters
struct ParameterEdge : public ParameterEdgeBase {
  explicit ParameterEdge(Parameters* p) : dim(p->dim), params(p) {}
  bool has_parameters() const override;
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  void accumulate_grad(const Tensor& g) override;
  Dim dim;
  Parameters* params;
};

// represents specified (not learned) inputs to the network
struct InputEdge : public Edge {
  explicit InputEdge(const Dim& d, const std::vector<float>* pd) : dim(d), pdata(pd) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  Dim dim;
  const std::vector<float>* pdata;
};

// represents specified (not learned) scalar inputs to the network
struct ScalarInputEdge : public Edge {
  explicit ScalarInputEdge(real s) : data(s), pdata(&data) {}
  explicit ScalarInputEdge(const real* ps) : data(), pdata(ps) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  const cnn::real data;
  const cnn::real* pdata;
};

// represents a matrix/vector embedding of an item of a discrete set (1-hot coding)
struct LookupEdge : public ParameterEdgeBase {
  LookupEdge(LookupParameters* p, unsigned ind) : dim(p->dim), index(ind), pindex(&index), params(p), has_optimizable_parameters(true) {}
  LookupEdge(LookupParameters* p, const unsigned* pind) : dim(p->dim), index(), pindex(pind), params(p), has_optimizable_parameters(true) {}
  bool has_parameters() const override;
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Tensor forward(const std::vector<const Tensor*>& xs) const override;
  Tensor backward(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i) const override;
  void accumulate_grad(const Tensor& g) override;
  Dim dim;
  unsigned index;
  const unsigned* pindex;
  LookupParameters* params;
  bool has_optimizable_parameters;
};

} // namespace cnn

#endif
