#ifndef CNN_NODES_CONV_H_
#define CNN_NODES_CONV_H_

#include "cnn/cnn.h"
#include "cnn/nodes-macros.h"

namespace cnn {

struct KMaxPooling : public Node {
  explicit KMaxPooling(const std::initializer_list<VariableIndex>& a, unsigned k = 1) : Node(a), k(k) {}
  size_t aux_storage_size() const override;
  CNN_NODE_DEFINE_DEV_IMPL()
  unsigned k;
};

struct FoldRows : public Node {
  explicit FoldRows(const std::initializer_list<VariableIndex>& a, unsigned nrows) : Node(a), nrows(nrows) {}
  CNN_NODE_DEFINE_DEV_IMPL()
  unsigned nrows;
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DNarrow : public Node {
  explicit Conv1DNarrow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  CNN_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DWide : public Node {
  explicit Conv1DWide(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  CNN_NODE_DEFINE_DEV_IMPL()
};

} // namespace cnn

#endif
