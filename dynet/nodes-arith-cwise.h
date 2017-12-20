#ifndef DYNET_NODES_ARITH_CWISE_H_
#define DYNET_NODES_ARITH_CWISE_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = \sum_i x_i
struct CwiseSum : public Node {
  template <typename T> explicit CwiseSum(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
  template<class MyDevice, int ReductionOrder>
  void backward_helper(const MyDevice & dev,
					   const std::vector<const Tensor*>& xs,
					   const Tensor& fx,
					   const Tensor& dEdf,
					   unsigned i,
					   Tensor& dEdxi) const;
};


// y = x_1 \cdot x_2  (Hadamard product)
struct CwiseMultiply : public Node {
  explicit CwiseMultiply(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
  template<class MyDevice, int ReductionOrder>
  void backward_helper(const MyDevice & dev,
		                             const std::vector<const Tensor*>& xs,
		                             const Tensor& fx,
		                             const Tensor& dEdf,
		                             unsigned i,
		                             Tensor& dEdxi) const;
};

// y = x_1 / x_2  (cwiseQuotient)
struct CwiseQuotient : public Node {
  explicit CwiseQuotient(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  template<class MyDevice, int ReductionOrder>
  void backward_helper(const MyDevice & dev,
		       const std::vector<const Tensor*>& xs,
		       const Tensor& fx,
		       const Tensor& dEdf,
		       unsigned i,
		       Tensor& dEdxi) const;
};

// y = pow(x_1, x_2)
// x_2 raise every element in x_1 to the power of scalar x_2
struct Pow : public Node {
  explicit Pow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

} // namespace dynet

#endif
