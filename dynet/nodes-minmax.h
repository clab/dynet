#ifndef DYNET_NODES_MINMAX_H_
#define DYNET_NODES_MINMAX_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = min{x_1, x_2}
struct Min : public Node {
  explicit Min(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
};

// y = max{x_1, x_2}
struct Max : public Node {
  template <typename T> explicit Max(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
};

struct MinDimension : public Node {
  explicit MinDimension(const std::initializer_list<VariableIndex>& a, unsigned dimension = 0) : Node(a), reduced_dim(dimension) {
    first_dim = reduced_dim == 0 ? 1 : 0;
    second_dim = first_dim + 1 == reduced_dim ? first_dim + 2 : first_dim + 1;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  unsigned reduced_dim;
  unsigned first_dim;
  unsigned second_dim;
};

struct MaxDimension : public Node {
  explicit MaxDimension(const std::initializer_list<VariableIndex>& a, unsigned dimension = 0) : Node(a), reduced_dim(dimension) {
    first_dim = reduced_dim == 0 ? 1 : 0;
    second_dim = first_dim + 1 == reduced_dim ? first_dim + 2 : first_dim + 1;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  unsigned reduced_dim;
  unsigned first_dim;
  unsigned second_dim;
};

} // namespace dynet

#endif
