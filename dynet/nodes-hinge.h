#ifndef DYNET_NODES_HINGE_H_
#define DYNET_NODES_HINGE_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

// Let x be a std::vector-valued input, x_i represents the score of the ith element, then
// y = \sum{i != element} max{0, margin - x_element + x_i}
struct Hinge : public Node {
  explicit Hinge(const std::initializer_list<VariableIndex>& a, unsigned e, real m = 1.0) : Node(a), element(e), pelement(&element), margin(m) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const unsigned* pe, real m = 1.0) : Node(a), element(), pelement(pe), margin(m) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& e, real m = 1.0) : Node(a), element(), pelement(), elements(e), pelements(&elements), margin(m) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pe, real m = 1.0) : Node(a), element(), pelement(), elements(), pelements(pe), margin(m) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  unsigned element;
  const unsigned* pelement;
  std::vector<unsigned> elements;
  const std::vector<unsigned>* pelements;
  real margin;
};

} // namespace dynet

#endif
