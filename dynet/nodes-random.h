#ifndef DYNET_NODES_RANDOM_H_
#define DYNET_NODES_RANDOM_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// n_{i,j} ~ N(0,stddev)
// y = x + n
struct GaussianNoise : public Node {
  explicit GaussianNoise(const std::initializer_list<VariableIndex>& a, real stddev) : Node(a), stddev(stddev) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  real stddev;
};

// draw random noise from Normal(0, 1)
struct RandomNormal : public Node {
  explicit RandomNormal(const Dim& d, float m=0.f, float s=1.f) : dim(d), mean(m), stddev(s) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  float mean, stddev;
};

// draw from Bernoulli(p)
struct RandomBernoulli : public Node {
  explicit RandomBernoulli(const std::initializer_list<VariableIndex>& a, const Dim& d, real p, real scale = 1.0f) : dim(d), p(p), scale(scale) {
    DYNET_ASSERT(a.size() == 0, "RandomBernoulli doesn't accept nodes as input");
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  real p;
  real scale;
};

// draw a random real from Uniform(left, right)
struct RandomUniform : public Node {
  explicit RandomUniform(const std::initializer_list<VariableIndex>& a, const Dim& d, real left, real right) : dim(d), left(left), right(right) {
    DYNET_ASSERT(a.size() == 0, "RandomUniform doesn't accept nodes as input");
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  real left, right;
};

// draw a random real from Uniform(left, right)
struct RandomGumbel : public Node {
  explicit RandomGumbel(const std::initializer_list<VariableIndex>& a, const Dim& d, real mu, real beta) : dim(d), mu(mu), beta(beta) {
    DYNET_ASSERT(a.size() == 0, "RandomGumbel doesn't accept nodes as input");
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  real mu, beta;
};

} // namespace dynet

#endif
