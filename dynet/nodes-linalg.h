#ifndef DYNET_NODES_LINALG_H_
#define DYNET_NODES_LINALG_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = x_1^T
// NOTE: if you have a column or row std::vector as input, runtime is constant
// if you have a matrix as input, the runtime is O(mn) - try to avoid using this
struct Transpose : public Node {
  explicit Transpose(const std::initializer_list<VariableIndex>& a,
                     const std::vector<unsigned>& dims)
      : Node(a), dims(dims) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                 const std::vector<VariableIndex> & batch_ids,
                                 const std::vector<int> & concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  std::vector<unsigned> dims;
};

// y = inv(x)
// x = an invertible matrix
struct MatrixInverse : public Node {
  explicit MatrixInverse(const std::initializer_list<VariableIndex>& a)
      : Node(a) {
    this->has_cuda_implemented = false;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = log det(x)
struct LogDet : public Node {
  template <typename T>
  explicit LogDet(const T& a) : Node(a) {
    this->has_cuda_implemented = false;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = Tr(x_1 * x_2^T)
struct TraceOfProduct : public Node {
  explicit TraceOfProduct(const std::initializer_list<VariableIndex>& a)
      : Node(a) {
    this->has_cuda_implemented = false;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

}  // namespace dynet

#endif
