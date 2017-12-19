#ifndef DYNET_NODES_CONV_H_
#define DYNET_NODES_CONV_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// y = x_1 *filter x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Filter1DNarrow : public Node {
  explicit Filter1DNarrow(const std::initializer_list<VariableIndex>& a)
      : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

struct FoldRows : public Node {
  explicit FoldRows(const std::initializer_list<VariableIndex>& a,
                    unsigned nrows)
      : Node(a), nrows(nrows) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned nrows;
};

struct KMaxPooling : public Node {
  explicit KMaxPooling(const std::initializer_list<VariableIndex>& a,
                       unsigned k = 1, unsigned dimension = 1)
      : Node(a), k(k), pooled_dim(dimension) {
    first_dim = pooled_dim == 0 ? 1 : 0;
    second_dim = first_dim + 1 == pooled_dim ? first_dim + 2 : first_dim + 1;
  }
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned k;
  unsigned pooled_dim;
  unsigned first_dim;
  unsigned second_dim;
};

// y_i = \sum_{j=1}^n x_1:{i-1+j}
struct KMHNGram : public Node {
  explicit KMHNGram(const std::initializer_list<VariableIndex>& a, unsigned n)
      : Node(a), n(n) {
    this->has_cuda_implemented = false;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned n;  // width, n=2 for Karl's paper
};

// hyperparameter: width > 1
// x_1 is a std::vector in R^n, which we write x
// y is a std::vector in R^{n / width}
// y_i = max_{x_{i * width - width + 1}, ..., x_{i * width}}
struct MaxPooling1D : public Node {
  MaxPooling1D(const std::initializer_list<VariableIndex>& a, unsigned w)
      : Node(a), width(w) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned width;
  mutable std::vector<unsigned> ind;
};

struct CircularConvolution : public Node {
  CircularConvolution(const std::initializer_list<VariableIndex>& a)
      : Node(a) {}
  size_t aux_storage_size() const override;
  DYNET_NODE_DEFINE_DEV_IMPL()
};

struct CircularCorrelation : public Node {
  CircularCorrelation(const std::initializer_list<VariableIndex>& a)
      : Node(a) {}
  size_t aux_storage_size() const override;
  DYNET_NODE_DEFINE_DEV_IMPL()
};

}  // namespace dynet

#endif
