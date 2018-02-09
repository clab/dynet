#ifndef DYNET_NODES_LOSSES_H_
#define DYNET_NODES_LOSSES_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// x_1 is a scalar (or row std::vector)
// x_2 is a scalar (or row std::vector)
// y = max(0, margin - x_1 + x_2)
struct PairwiseRankLoss : public Node {
  explicit PairwiseRankLoss(const std::initializer_list<VariableIndex>& a, real m = 1.0) : Node(a), margin(m) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  real margin;
};

// you could do this with LogisticSigmoid, Softmax or a variety of other
// functions, but this is often useful.
// x_1 must be a std::vector with values between 0 and 1
// target_y is an equivalently sized std::vector w values between 0 and 1
// y = ty * log(x_1) + (1 - ty) * log(x_1)
struct BinaryLogLoss : public Node {
  BinaryLogLoss(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// this is used to implement poisson regression
// x_1 = log predicted mean
// ty = true y (this is not a VariableIndex since it has to be a nonnegative integer and
//              is therefore nondifferentiable. There are various continuous extensions
//              using the incomplete gamma function that could be used, but meh)
// y = log Poisson(ty; \lambda = \exp x_1)
//   = ty*x_1 - exp(x_1) - log(ty!)
struct PoissonRegressionLoss : public Node {
  explicit PoissonRegressionLoss(const std::initializer_list<VariableIndex>& a, unsigned true_y) : Node(a), ty(true_y), pty(&ty) {}
  explicit PoissonRegressionLoss(const std::initializer_list<VariableIndex>& a, const unsigned* ptrue_y) : Node(a), ty(), pty(ptrue_y) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
 private:
  unsigned ty;
  const unsigned* pty;
};

} // namespace dynet

#endif
