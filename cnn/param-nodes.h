#ifndef CNN_PARAM_NODES_H_
#define CNN_PARAM_NODES_H_

#include "cnn/cnn.h"
#include "cnn/model.h"
#include "cnn/node-macros.h"

namespace cnn {

struct ParameterNodeBase : public Node {
  virtual void accumulate_grad(const Tensor& g) = 0;
};

// represents optimizable parameters
struct ParameterNode : public ParameterNodeBase {
  explicit ParameterNode(Parameter p) : dim(p.get()->dim), params(p) {}
  CNN_NODE_DEFINE_DEV_IMPL()
  void accumulate_grad(const Tensor& g) override;
  Dim dim;
  Parameter params;
};

// represents optimizable parameters that are being held constant
struct ConstParameterNode : public Node {
  explicit ConstParameterNode(Parameter p) : dim(p.get()->dim), params(p) {}
  CNN_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  Parameter params;
};

// represents specified (not learned) inputs to the network
struct InputNode : public Node {
  explicit InputNode(const Dim& d, const std::vector<float>& dat) : dim(d), data(dat), pdata(&data) {}
  explicit InputNode(const Dim& d, const std::vector<float>* pdat) : dim(d), data(), pdata(pdat) {}
  CNN_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  Dim dim;
  const std::vector<float> data;
  const std::vector<float>* pdata;
};

// represents specified (not learned) scalar inputs to the network
struct ScalarInputNode : public Node {
  explicit ScalarInputNode(real s) : data(s), pdata(&data) {}
  explicit ScalarInputNode(const real* ps) : data(), pdata(ps) {}
  CNN_NODE_DEFINE_DEV_IMPL()
  const cnn::real data;
  const cnn::real* pdata;
};

// represents a matrix/vector embedding of an item of a discrete set (1-hot coding)
struct LookupNode : public ParameterNodeBase {
  LookupNode(LookupParameter p, unsigned ind) : dim(p.get()->dim), index(ind), pindex(&index), indices(), pindices(), params(p) {}
  LookupNode(LookupParameter p, const unsigned* pind) : dim(p.get()->dim), index(), pindex(pind), indices(), pindices(), params(p) {}
  LookupNode(LookupParameter p, const std::vector<unsigned>& indices) : dim(p.get()->dim), index(), pindex(), indices(indices), pindices(&this->indices), params(p) {
    dim.bd = pindices->size();
  }
  LookupNode(LookupParameter p, const std::vector<unsigned>* pindices) : dim(p.get()->dim), index(), pindex(), indices(), pindices(pindices), params(p) {
    dim.bd = pindices->size();
  }
  CNN_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }  
  void accumulate_grad(const Tensor& g) override;
  Dim dim;
  unsigned index;
  const unsigned* pindex;
  std::vector<unsigned> indices;
  const std::vector<unsigned>* pindices;
  LookupParameter params;
};

} // namespace cnn

#endif
