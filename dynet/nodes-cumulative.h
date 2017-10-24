#ifndef DYNET_NODES_CUMULATIVE_H_
#define DYNET_NODES_CUMULATIVE_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {


//y_i = \sum_{j<i} x_j
struct CumulativeSum : public Node {
  template <typename T> explicit CumulativeSum(const T& a, unsigned d) : Node(a), d(d){}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  unsigned d;
};
}
#endif
