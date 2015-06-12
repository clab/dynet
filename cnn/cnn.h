#ifndef CNN_CNN_H_
#define CNN_CNN_H_

#include <string>
#include <vector>
#include <iostream>
#include <initializer_list>
#include <utility>
#include <boost/serialization/strong_typedef.hpp>

#include "cnn/init.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/tensor.h"
#include "cnn/model.h"

// Computation graph where nodes represent forward and backward intermediate
// values, and edges represent functions of multiple values. To represent the
// fact that a function may have multiple arguments, edges have a single head
// and 0, 1, 2, or more tails. (Constants, inputs, and parameters are
// represented as functions of 0 parameters.)
// Example: given the function z = f(x, y), z, x, and y are nodes, and there
// is an edge representing f with which points to the z node (i.e., its head),
// and x and y are the tails of the edge.

namespace cnn {

extern AlignedMemoryPool<5>* fxs;
extern AlignedMemoryPool<5>* dEdfs;
extern float* kSCALAR_MINUSONE;
extern float* kSCALAR_ONE;
extern float* kSCALAR_ZERO;

class ExecutionEngine;
struct ParameterNodeBase;
struct Node;

BOOST_STRONG_TYPEDEF(unsigned, VariableIndex)
inline void swap(VariableIndex& i1, VariableIndex& i2) {
  VariableIndex t = i1;
  i1 = i2;
  i2 = t;
}

struct ComputationGraph {
  ComputationGraph();
  ~ComputationGraph();

  // INPUTS
  // the computational network will pull inputs in from the user's data
  // structures and make them available to the computation
  VariableIndex add_input(real s);  // add scalar
  VariableIndex add_input(const real* ps);  // add pointer to scalar
  VariableIndex add_input(const Dim& d, const std::vector<float>* pdata);

  // PARAMETERS
  // parameters are things that are optimized. in contrast to a system like
  // Torch where computational modules may have their own parameters, in CNN
  // parameters are just parameters
  VariableIndex add_parameters(Parameters* p);
  // use pindex to point to a memory location where the index will live
  // that the caller owns
  VariableIndex add_lookup(LookupParameters* p, const unsigned* pindex);
  VariableIndex add_lookup(LookupParameters* p, unsigned index);
  // just like add_lookup, but don't optimize the lookup parameters
  VariableIndex add_const_lookup(LookupParameters* p, const unsigned* pindex);
  VariableIndex add_const_lookup(LookupParameters* p, unsigned index);

  // COMPUTATIONS
  template <class Function> inline VariableIndex add_function(const std::initializer_list<VariableIndex>& arguments);
  template <class Function, typename... Args>
  inline VariableIndex add_function(const std::initializer_list<VariableIndex>& arguments,
                                    Args&&... side_information);
  template <class Function, typename T> inline VariableIndex add_function(const T& arguments);

  // perform computations
  const Tensor& forward();
  const Tensor& incremental_forward();  // if you want to add nodes and evaluate just the new parts
  void backward();

  // debugging
  void PrintGraphviz() const;

  // data
  std::vector<Node*> nodes;       // **stored in topological order**
  std::vector<VariableIndex> parameter_nodes; // nodes that contain parameters that can be updated (subset of nodes)
  VariableIndex last_node_evaluated; // enables forward graphs to be evaluated incrementally

  ExecutionEngine* ee;  // handles the execution
 private:
  void set_dim_for_new_node(const VariableIndex& i);
};

// represents an SSA variable
// * in_edge is the **ordered** list of indices of the function arguments
// * fx is the computed value of the variable
// * dEdf is the derivative of the output with respect to the function
struct Node {
  virtual ~Node();

  // compute dimensions of result for given dimensions of inputs
  // also checks to make sure inputs are compatible with each other
  virtual Dim dim_forward(const std::vector<Dim>& xs) const = 0;

  // for debugging
  virtual std::string as_string(const std::vector<std::string>& args) const = 0;

  // in general, this will return an empty size, but if a component needs to store
  // extra information in the forward pass for use in the backward pass, it can
  // request the memory here (nb. you could put it on the Node object, but in general,
  // edges should not allocate tensor memory since memory is managed centrally for the
  // entire computation graph).
  virtual size_t aux_storage_size() const;

  // computation
  virtual void forward(const std::vector<const Tensor*>& xs,
                       Tensor& fx) const = 0;
  // accumulates the derivative of E with respect to the ith argument to f, that is, xs[i]
  virtual void backward(const std::vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const = 0;

  // number of arguments to the function
  inline unsigned arity() const { return args.size(); }

  // dependency structure
  std::vector<VariableIndex> args;

  // memory size
  Dim dim;  // will be .size() = 0 initially filled in by forward() -- TODO fix this

 protected:
  Node() : args() {}
  explicit Node(const std::initializer_list<VariableIndex>& a) : args(a) {}
  template <typename T>
  explicit Node(const T&c) : args(c.begin(), c.end()) {}

 public:
  // auxiliary memory
  mutable void* aux_mem; // this will usually be null. but, if your node needs to store intermediate values
                 // between forward and backward, you can use store it here. request the
                 // number of bytes you need from aux_storage_size(). Note:
                 // this memory will be on the CPU or GPU, depending on your computation
                 // backend
};

template <class Function>
inline VariableIndex ComputationGraph::add_function(const std::initializer_list<VariableIndex>& arguments) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Function(arguments));
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}

// pass side information to the function. these are likely to be nondifferentiable arguments
template <class Function, typename... Args>
inline VariableIndex ComputationGraph::add_function(const std::initializer_list<VariableIndex>& arguments,
                                              Args&&... side_information) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Function(arguments, std::forward<Args>(side_information)...));
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}

template <class Function, typename T>
inline VariableIndex ComputationGraph::add_function(const T& arguments) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Function(arguments));
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}

} // namespace cnn

#endif
