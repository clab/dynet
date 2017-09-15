#ifndef DYNET_NODES_LOGSUMEXP_H_
#define DYNET_NODES_LOGSUMEXP_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

// y = \log \sum_i \exp x_i
// done in log space carefully to avoid over/underflow issues
struct LogSumExp : public Node {
  template <typename T> explicit LogSumExp(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

} // namespace dynet

#endif
