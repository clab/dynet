#ifndef DYNET_NODES_DROPOUT_H_
#define DYNET_NODES_DROPOUT_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = dropout(x,p) where p specifies the dropout probability
struct Dropout : public Node {
  explicit Dropout(const std::initializer_list<VariableIndex>& a, real p) : Node(a), p(p) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  real p;
};

// y = dropout(x,p) where p specifies the dropout probability
struct DropoutDim : public Node {
  explicit DropoutDim(const std::initializer_list<VariableIndex>& a, unsigned d,real p) : Node(a), dimension(d), p(p) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  unsigned dimension;
  real p;
};

// y = dropout(x,p) where p specifies the dropout probability
struct DropoutBatch : public Node {
  explicit DropoutBatch(const std::initializer_list<VariableIndex>& a, real p) : Node(a), p(p) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  real p;
};

// y = block_dropout(x,p) where p specifies the probability for dropping-out the entire block
struct BlockDropout : public Node {
  explicit BlockDropout(const std::initializer_list<VariableIndex>& a, real p) : Node(a), dropout_probability(p) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  real dropout_probability;
};

} // namespace dynet

#endif
