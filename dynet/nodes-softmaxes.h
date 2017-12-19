#ifndef DYNET_NODES_SOFTMAXES_H_
#define DYNET_NODES_SOFTMAXES_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// z = \sum_j \exp (x_i)_j
// y_i = (x_1)_i / z
struct Softmax : public Node {
  explicit Softmax(const std::initializer_list<VariableIndex>& a, unsigned d=0) : Node(a), dimension(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph& cg,
                            SigMap& sm) const override;
  virtual std::vector<int> autobatch_concat(
      const ComputationGraph& cg) const override;
  virtual void autobatch_reshape(const ComputationGraph& cg,
                                 const std::vector<VariableIndex>& batch_ids,
                                 const std::vector<int>& concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  unsigned dimension;
};

// z = \sum_j \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct LogSoftmax : public Node {
  explicit LogSoftmax(const std::initializer_list<VariableIndex>& a)
      : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
};

// z = \sum_{j \in denom} \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct RestrictedLogSoftmax : public Node {
  explicit RestrictedLogSoftmax(const std::initializer_list<VariableIndex>& a,
                                const std::vector<unsigned>& d)
      : Node(a), denom(d) {
    this->has_cuda_implemented = false;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  std::vector<unsigned> denom;
};

// y = sparsemax(x)
// y = arg min_y ||y - x||^2
struct Sparsemax : public Node {
  explicit Sparsemax(const std::initializer_list<VariableIndex>& a) : Node(a) {
    this->has_cuda_implemented = false;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
};

// y = L_sparsemax(x_0; q)
// where x_0 is a std::vector of "unnormalized" probabilities
// q are the std::vector of labels
struct SparsemaxLoss : public Node {
  explicit SparsemaxLoss(const std::initializer_list<VariableIndex>& a,
                         const std::vector<unsigned>& target)
      : Node(a), q(target), pq(&q) {
    this->has_cuda_implemented = false;
  }
  explicit SparsemaxLoss(const std::initializer_list<VariableIndex>& a,
                         const std::vector<unsigned>* ptarget)
      : Node(a), q(), pq(ptarget) {
    this->has_cuda_implemented = false;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  const std::vector<unsigned> q;
  const std::vector<unsigned>* pq;
};

// y = constrained_softmax(x, u)
// y = arg min_{y<=u} KL(y || x)
struct ConstrainedSoftmax : public Node {
  explicit ConstrainedSoftmax(const std::initializer_list<VariableIndex>& a)
      : Node(a) {
    this->has_cuda_implemented = false;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
};

}  // namespace dynet

#endif
