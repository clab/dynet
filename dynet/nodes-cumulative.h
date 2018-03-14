#ifndef DYNET_NODES_CUMULATIVE_H_
#define DYNET_NODES_CUMULATIVE_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {


//y_i = \sum_{j<i} x_j
struct CumulativeSum : public Node {
  template <typename T> explicit CumulativeSum(const T& a, unsigned d) : Node(a), d(d){}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
private:
  unsigned d;
};
}
#endif
