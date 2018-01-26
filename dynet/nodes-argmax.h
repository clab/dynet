#ifndef DYNET_NODES_ARGMAX_H_
#define DYNET_NODES_ARGMAX_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// z = \sum_j \exp (x_i)_j
// y = (x_1)_element - \log z
struct Argmax : public Node {
  explicit Argmax(const std::initializer_list<VariableIndex>& a, unsigned d) : Node(a), d(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  unsigned d;
};

} // namespace dynet

#endif
