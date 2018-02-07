#ifndef DYNET_NODES_ARGMAX_H_
#define DYNET_NODES_ARGMAX_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y_i = 1 if i = argmax(x) else 0
struct Argmax : public Node {
  explicit Argmax(const std::initializer_list<VariableIndex>& a, unsigned d, bool straight_through=false) : Node(a), d(d), straight_through(straight_through) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  unsigned d;
  bool straight_through;
};

} // namespace dynet

#endif
