#ifndef DYNET_NODES_CONV_H_
#define DYNET_NODES_CONV_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

// with a single argument x \in R^{n x m}
// y_i = \sum_j x_i,j / m
struct AverageColumns : public Node {
  template <typename T> explicit AverageColumns(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DNarrow : public Node {
  explicit Conv1DNarrow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DWide : public Node {
  explicit Conv1DWide(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 *filter x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Filter1DNarrow : public Node {
  explicit Filter1DNarrow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

struct FoldRows : public Node {
  explicit FoldRows(const std::initializer_list<VariableIndex>& a, unsigned nrows) : Node(a), nrows(nrows) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned nrows;
};

struct KMaxPooling : public Node {
  explicit KMaxPooling(const std::initializer_list<VariableIndex>& a, unsigned k = 1) : Node(a), k(k) {}
  size_t aux_storage_size() const override;
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned k;
};

// sum along a single dimension
struct SumDimension : public Node {
  template <typename T> explicit SumDimension(const T& a, unsigned d) : Node(a), dimension(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned dimension;
};

// 2D convolution
// TODO(Hao Zhang): move conv2d to standalone files because the code logic could be very long
// when cudnn is incorporated.
// y = x_1 *conv x_2
// x_1 \in R^{Ci x W x H x N} (input)
// x_2 \in R^{Co x Ci x W x H} (filter)
// stride[0] corresponds to H
// stride[1] corresponds to W
// is_valid: true for 'VALID' and false for 'SAME'
// Note: You may find the dimension here a bit counter-intuitive (e.g. W is ahead of H).
// The reasons are as follows: in an umcomming GPU implementation, cuDNN can only support NCHW 
// and NHWC formated Tensor. Follwing DyNet's convention using Eigen::ColMajor, the tensor can 
// only be viewed as D1 x D2 x D3 x N (the batchsize N is at the last dimension). As swapping
// dimension of a 4D tensor in runtime is computational prohibitive, I decide to arrange the 
// tensor as CWHN, so that when using Eigen::RowMajor, we will directly get a NHWC tensor, 
// which follows a standard cuDNN convention (Viewas<RowMajor> operation does not have cost).
// On the other hand, CPU version Eigen::SpatialConvolution takes input tensor in CHWN/NWHC format, 
// so there is another simple trick here: when calling Eigen::SpatialConvolution, swap the Col and 
// Row dim in the function argument list to fool this function -- you will still get a correct 
// result except the col and row are swapped.
struct Conv2D: public Node {
  explicit Conv2D(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& s,
    const bool padding_type = true)
      : Node(a), stride(s), is_valid(padding_type) { }
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  const std::vector<unsigned> stride;
  const bool is_valid;
};

} // namespace dynet

#endif
