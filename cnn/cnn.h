#ifndef CNN_CNN_H_
#define CNN_CNN_H_

#include <string>
#include <vector>
#include <iostream>
#include <initializer_list>
#include <utility>
#include <boost/serialization/strong_typedef.hpp>

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

typedef AlignedMemoryPool<5> MemoryPool; // ?
extern AlignedMemoryPool<5>* fxs;
extern AlignedMemoryPool<5>* dEdfs;

struct Edge;
struct ParameterEdgeBase;
struct Node;

BOOST_STRONG_TYPEDEF(unsigned, VariableIndex)
inline void swap(VariableIndex& i1, VariableIndex& i2) {
  VariableIndex t = i1;
  i1 = i2;
  i2 = t;
}

struct Hypergraph {
  Hypergraph();
  ~Hypergraph();

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
  VariableIndex add_parameter(Parameters* p);
  // use pindex to point to a memory location where the index will live
  // that the caller owns
  VariableIndex add_lookup(LookupParameters* p, const unsigned* pindex);
  VariableIndex add_lookup(LookupParameters* p, unsigned index);
  // just like add_lookup, but don't optimize the lookup parameters
  VariableIndex add_const_lookup(LookupParameters* p, unsigned* pindex);
  VariableIndex add_const_lookup(LookupParameters* p, unsigned index);

  // COMPUTATIONS
  template <class Function> inline VariableIndex add_function(const std::initializer_list<VariableIndex>& arguments);
  template <class Function, typename... Args>
  inline VariableIndex add_function(const std::initializer_list<VariableIndex>& arguments,
                                    Args&&... side_information);
  template <class Function, typename T> inline VariableIndex add_function(const T& arguments);

  // debugging
  void PrintGraphviz() const;

  // data
  std::vector<Node*> nodes;       // **stored in topological order**
  std::vector<Edge*> edges;       // all edges
  std::vector<ParameterEdgeBase*> parameter_edges; // edges that contain parameters that can be updated (subset of edges)
};

// represents an SSA variable
// * in_edge is the index of the function that computes the variable
// * out_edges are the list of functions that use this variable
// * f is the computed value of the variable (TODO: remove this, see note below)
// * dEdf is the derivative of the output with respect to the function
struct Node {
  // name is currently just used for debugging- maybe eventually for code
  // generation though
  Node(unsigned in_edge_index, VariableIndex nid) :
      in_edge(in_edge_index),
      node_id(nid) {}

  // dependency structure
  unsigned in_edge;
  std::vector<unsigned> out_edges;

  // debugging
  std::string variable_name() const { return "v" + std::to_string(node_id); }
  VariableIndex node_id;  // my id
};

inline void swap(Node& n1, Node& n2) {
  using std::swap;
  swap(n1.in_edge, n2.in_edge);
  swap(n1.out_edges, n2.out_edges);
  swap(n1.node_id, n2.node_id);
}

// represents a function of zero or more input variables
// functions with zero inputs are constants or optimizeable parameters
struct Edge {
  virtual ~Edge();
  // debugging
  virtual std::string as_string(const std::vector<std::string>& var_names) const = 0;

  // compute dimensions of result for given dimensions of inputs
  // also checks to make sure inputs are compatible with each other
  virtual Dim dim_forward(const std::vector<Dim>& xs) const = 0;

  // in general, this will return an empty size, but if a component needs to store
  // extra information in the forward pass for use in the backward pass, it can
  // request the memory here (nb. you could put it on the Edge object, but in general,
  // edges should not allocate tensor memory since memory is managed centrally for the
  // entire computation graph). TODO
  // virtual Dim aux_storage_space() const;

  // computation
  virtual void forward(const std::vector<const Tensor*>& xs,
                       Tensor& fx) const = 0;
  // computes the derivative of E with respect to the ith argument to f, that is, xs[i]
  virtual void backward(const std::vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const = 0;
  virtual bool has_parameters() const;

  // number of arguments to the function
  inline unsigned arity() const { return tail.size(); }

  // structure
  VariableIndex head_node;   // index of node to contain result of f
  std::vector<VariableIndex> tail;  // arguments of function
};

inline void swap(Edge& e1, Edge& e2) {
  using std::swap;
  swap(e1.tail, e2.tail);
  swap(e1.head_node, e2.head_node);
}

template <class Function>
inline VariableIndex Hypergraph::add_function(const std::initializer_list<VariableIndex>& arguments) {
  VariableIndex new_node_index(nodes.size());
  unsigned new_edge_index = edges.size();
  nodes.push_back(new Node(new_edge_index, new_node_index));
  Edge* new_edge = new Function;
  edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  for (auto ni : arguments) {
    new_edge->tail.push_back(ni);
    nodes[ni]->out_edges.push_back(new_edge_index);
  }
  return new_node_index;
}

// pass side information to the function. these are likely to be nondifferentiable arguments
template <class Function, typename... Args>
inline VariableIndex Hypergraph::add_function(const std::initializer_list<VariableIndex>& arguments,
                                              Args&&... side_information) {
  VariableIndex new_node_index(nodes.size());
  unsigned new_edge_index = edges.size();
  nodes.push_back(new Node(new_edge_index, new_node_index));
  Edge* new_edge = new Function(std::forward<Args>(side_information)...);
  edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  for (auto ni : arguments) {
    new_edge->tail.push_back(ni);
    nodes[ni]->out_edges.push_back(new_edge_index);
  }
  return new_node_index;
}

template <class Function, typename T>
inline VariableIndex Hypergraph::add_function(const T& arguments) {
  VariableIndex new_node_index(nodes.size());
  unsigned new_edge_index = edges.size();
  nodes.push_back(new Node(new_edge_index, new_node_index));
  Edge* new_edge = new Function;
  edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  for (auto ni : arguments) {
    new_edge->tail.push_back(ni);
    nodes[ni]->out_edges.push_back(new_edge_index);
  }
  return new_node_index;
}

struct RunNode;

struct Run {
  Run(Hypergraph *hg, MemoryPool *fxs, MemoryPool *dEdfs) 
  : hg(hg), 
    fxs(fxs), 
    dEdfs(dEdfs),
    last_node_evaluated(),
    runnodes()
  { };

  Hypergraph *hg;
  MemoryPool *fxs, *dEdfs;

  VariableIndex last_node_evaluated; // enables forward graphs to be evaluated incrementally

  std::vector<RunNode> runnodes; // parallel to Hypergraph::nodes

  // perform computations
  const Tensor& forward();
  const Tensor& incremental_forward();  // if you want to add nodes and evaluate just the new parts
  void backward();
};

// All the information about a node that pertains to a particular run
struct RunNode {
  // memory
  Dim dim;  // will be .size() = 0 initially, before memory is allocated

  // computation results (nb. memory is not owned by Tensor)
  Tensor f;               // f(x_1 , ... , x_n)
  Tensor dEdf;            // dE/df
};

inline void swap(RunNode& n1, RunNode& n2) {
  using std::swap;
  swap(n1.dim, n2.dim);
  swap(n1.f, n2.f);
  swap(n1.dEdf, n2.dEdf);
}

} // namespace cnn

#endif
