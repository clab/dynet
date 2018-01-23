#ifndef DYNET_NODES_MAXPOOLING2D_H_
#define DYNET_NODES_MAXPOOLING2D_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

#if HAVE_CUDNN
#include "dynet/cudnn-ops.h"
#endif

namespace dynet {

// maxpooling2d
// y = x_1 * maxpooling2d
// x_1 \in R^{H x W x Ci x N} (input)
// ksize[0] corresponds to H
// ksize[1] corresponds to W
// stride[0] corresponds to H
// stride[1] corresponds to W
// is_valid: true for 'VALID' and false for 'SAME'
struct MaxPooling2D: public Node {
  explicit MaxPooling2D(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& k, const std::vector<unsigned>& s,
    const bool padding_type = true)
      : Node(a), ksize(k), stride(s), is_valid(padding_type) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  const std::vector<unsigned> ksize;
  const std::vector<unsigned> stride;
  const bool is_valid;

 private:
#if HAVE_CUDNN
  mutable CudnnMaxPooling2DOp* cudnn_maxpool_op_ = NULL;
#endif
};

} // namespace dynet

#endif
