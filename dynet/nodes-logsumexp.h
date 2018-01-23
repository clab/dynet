#ifndef DYNET_NODES_LOGSUMEXP_H_
#define DYNET_NODES_LOGSUMEXP_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = \log \sum_i \exp x_i
// done in log space carefully to avoid over/underflow issues
struct LogSumExp : public Node {
  template <typename T> explicit LogSumExp(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

struct LogSumExpDimension : public Node {
  template <typename T> explicit LogSumExpDimension(const T& a, unsigned d = 0) : Node(a), dimension(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  unsigned dimension;
};

} // namespace dynet

#endif
