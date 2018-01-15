#ifndef DYNET_NODES_HINGE_H_
#define DYNET_NODES_HINGE_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"

namespace dynet {

// Let x be a vector-valued input, x_i represents the score of the ith element, then
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

// Let x be a matrix input. This will calculate the loss over all rows or columns.
struct HingeDim : public Node {
  explicit HingeDim(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& e, unsigned d = 0, real m = 1.0) : Node(a), element(e), pelement(&element), pelements(nullptr), d(d), margin(m), input_size(0) {}
  explicit HingeDim(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pe, unsigned d = 0, real m = 1.0) : Node(a), element(), pelement(pe), pelements(nullptr), d(d), margin(m), input_size(0) {}
  explicit HingeDim(const std::initializer_list<VariableIndex>& a, const std::vector<std::vector<unsigned> >& e, unsigned d = 0, real m = 1.0) : Node(a), element(), pelement(), elements(e), pelements(&elements), d(d), margin(m), input_size(0) {}
  explicit HingeDim(const std::initializer_list<VariableIndex>& a, const std::vector<std::vector<unsigned> >* pe, unsigned d = 0, real m = 1.0) : Node(a), element(), pelement(), elements(), pelements(pe), d(d), margin(m), input_size(0) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  std::vector<unsigned> element;
  const std::vector<unsigned>* pelement;
  std::vector<std::vector<unsigned> > elements;
  const std::vector<std::vector<unsigned> >* pelements;
  unsigned d;
  real margin;
  size_t input_size;
};

} // namespace dynet

#endif
