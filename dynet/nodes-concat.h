#ifndef DYNET_NODES_CONCAT_H_
#define DYNET_NODES_CONCAT_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// concatenate along a particular dimension
struct Concatenate : public Node {
  template <typename T> explicit Concatenate(const T& a, unsigned d) : Node(a), dimension(d) {}
  virtual bool supports_multibatch() const override { return true; }
  virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
  virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override { return std::vector<int>(args.size(), 1); }  
  virtual void autobatch_reshape(const ComputationGraph & cg,
                                 const std::vector<VariableIndex> & batch_ids,
                                 const std::vector<int> & concat,
                                 std::vector<const Tensor*>& xs,
                                 Tensor& fx) const override {
    autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  // src_row_indices[i] says what row in fx the ith x std::vector was assigned to
  // used to simplify backprop
  mutable std::vector<unsigned> src_indices;
  unsigned dimension;
};

// concatenate different batched experssions into one single batched tensor
struct ConcatenateToBatch : public Node {
  template <typename T> explicit ConcatenateToBatch(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override {return true;}
  mutable std::vector<unsigned> src_element_indices;
};

} // namespace dynet

#endif
