#ifndef DYNET_NODES_CONV2D_H_
#define DYNET_NODES_CONV2D_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

#if HAVE_CUDNN
#include "dynet/cudnn-ops.h"
#endif

namespace dynet {

// conv2d 
// y = x_1 *conv2d x_2
// x_1 \in R^{H x W x Ci x N} (input)
// x_2 \in R^{H x W x Ci x Co} (filter)
// stride[0] corresponds to H
// stride[1] corresponds to W
// is_valid: true for 'VALID' and false for 'SAME'
struct Conv2D: public Node {
  explicit Conv2D(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& s,
    const bool padding_type = true)
      : Node(a), stride(s), is_valid(padding_type) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                 const std::vector<VariableIndex> & batch_ids,
                                 const std::vector<int> & concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  const std::vector<unsigned> stride;
  const bool is_valid;

 private:
#if HAVE_CUDNN
  mutable CudnnConvOp* cudnn_conv_op_ = NULL;
#endif
};

} // namespace dynet

#endif
