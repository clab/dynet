#ifndef CNN_PARAM_NODES_H_
#define CNN_PARAM_NODES_H_

#include "cnn/cnn.h"
#include "cnn/model.h"

namespace cnn {

struct ParameterNodeBase : public Node {
  virtual void accumulate_grad(const Tensor& g) = 0;
};

// represents optimizable parameters
struct ParameterNode : public ParameterNodeBase {
  explicit ParameterNode(Parameters* p) : dim(p->dim), params(p) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  void accumulate_grad(const Tensor& g) override;
  Dim dim;
  Parameters* params;
};

// represents optimizable parameters that are being held constant
struct ConstParameterNode : public Node {
  explicit ConstParameterNode(Parameters* p) : dim(p->dim), params(p) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  Dim dim;
  Parameters* params;
};

// represents specified (not learned) inputs to the network
struct InputNode : public Node {
  explicit InputNode(const Dim& d, const std::vector<float>& dat) : dim(d), data(dat), pdata(&data) {}
  explicit InputNode(const Dim& d, const std::vector<float>* pdat) : dim(d), data(), pdata(pdat) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  Dim dim;
  const std::vector<float> data;
  const std::vector<float>* pdata;
};

// represents specified (not learned) scalar inputs to the network
struct ScalarInputNode : public Node {
  explicit ScalarInputNode(real s) : data(s), pdata(&data) {}
  explicit ScalarInputNode(const real* ps) : data(), pdata(ps) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  const cnn::real data;
  const cnn::real* pdata;
};

// represents a matrix/vector embedding of an item of a discrete set (1-hot coding)
struct LookupNode : public ParameterNodeBase {
  LookupNode(LookupParameters* p, unsigned ind) : dim(p->dim), index(ind), pindex(&index), indices(), pindices(), params(p) {}
  LookupNode(LookupParameters* p, const unsigned* pind) : dim(p->dim), index(), pindex(pind), indices(), pindices(), params(p) {}
  LookupNode(LookupParameters* p, const std::vector<unsigned>& indices) : dim(p->dim), index(), pindex(), indices(indices), pindices(&this->indices), params(p) {
    dim.bd = pindices->size();
  }
  LookupNode(LookupParameters* p, const std::vector<unsigned>* pindices) : dim(p->dim), index(), pindex(), indices(), pindices(pindices), params(p) {
    dim.bd = pindices->size();
  }
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  virtual bool supports_multibatch() const override { return true; }  
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const override;
  void accumulate_grad(const Tensor& g) override;
  Dim dim;
  unsigned index;
  const unsigned* pindex;
  std::vector<unsigned> indices;
  const std::vector<unsigned>* pindices;
  LookupParameters* params;
};

} // namespace cnn

#endif
