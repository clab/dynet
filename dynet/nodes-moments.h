#ifndef DYNET_NODES_MOMENTS_H_
#define DYNET_NODES_MOMENTS_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = ( \sum_i x_i ) / |x|
struct Average : public Node {
  template <typename T> explicit Average(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

// with a single argument x \in R^{n x m}
// y_i = \sum_j x_i,j / m
struct AverageColumns : public Node {
  template <typename T> explicit AverageColumns(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = \sum_i,j,... x[i,j,...]
struct MomentElements : public Node {
  template <typename T> explicit MomentElements(const T& a, unsigned o) : Node(a), order(o) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  unsigned order;
};

// y = \sum_i x_i
struct MomentBatches : public Node {
  template <typename T> explicit MomentBatches(const T& a, unsigned o) : Node(a), order(o) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  unsigned order;
};

//y = \sum_i x_i
struct MomentDimension : public Node {
  template <typename T> explicit MomentDimension(const T& a, const std::vector<unsigned> & d, unsigned o, bool b=false, unsigned n=0) : Node(a), dims(d), order(o), include_batch_dim(b), overwrite_n(n) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  std::vector<unsigned> dims;
  unsigned order;
  bool include_batch_dim;
  unsigned overwrite_n;
};

// y = \sum_i,j,... x[i,j,...]
struct StdElements : public Node {
  template <typename T> explicit StdElements(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

// y = \sum_i x_i
struct StdBatches : public Node {
  template <typename T> explicit StdBatches(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

//y = \sum_i x_i
struct StdDimension : public Node {
  template <typename T> explicit StdDimension(const T& a, const std::vector<unsigned> & d, bool b=false, unsigned n=0) : Node(a), dims(d), include_batch_dim(b), overwrite_n(n) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  std::vector<unsigned> dims;
  bool include_batch_dim;
  unsigned overwrite_n;
};

} // namespace dynet

#endif
