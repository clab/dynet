#ifndef DYNET_NODES_HINGE_H_
#define DYNET_NODES_HINGE_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

// Let x be a std::vector-valued input, x_i represents the score of the ith element, then
// y = \sum{i != element} max{0, margin - x_element + x_i}
struct Hinge : public Node {
  explicit Hinge(const std::initializer_list<VariableIndex>& a, unsigned e, real m = 1.0) : Node(a), element(e), pelement(&element), margin(m), input_size(0) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const unsigned* pe, real m = 1.0) : Node(a), element(), pelement(pe), margin(m), input_size(0) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& e, real m = 1.0) : Node(a), element(), pelement(), elements(e), pelements(&elements), margin(m), input_size(0) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pe, real m = 1.0) : Node(a), element(), pelement(), elements(), pelements(pe), margin(m), input_size(0) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  unsigned element;
  const unsigned* pelement;
  std::vector<unsigned> elements;
  const std::vector<unsigned>* pelements;
  real margin;
  size_t input_size;
};

// // x can be either a matrix (or a batch of vectors where the batch size is equal to the vector size).
// // This will calculate a hinge loss in both directions: 
// // y = \sum_i \sum_{j != i} max{0, margin - x_i,i + x_i,j} + max{0, margin - x_j,j + x_i,j}
// struct BidirectionalHinge : public Node {
//   explicit BidirectionalHinge(const std::initializer_list<VariableIndex>& a, real m = 1.0) : Node(a), margin(m), input_size(0) {}
//   virtual bool supports_multibatch() const override { return true; }
//   DYNET_NODE_DEFINE_DEV_IMPL()
//   size_t aux_storage_size() const override;
//   real margin;
//   size_t input_size;
// };

} // namespace dynet

#endif
