#ifndef CNN_CNN_H_
#define CNN_CNN_H_

#include <string>
#include <vector>
#include <iostream>
#include <initializer_list>
#include <utility>
#include <Eigen/Eigen>
#include <boost/serialization/strong_typedef.hpp>

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

// TODO pull fx and dEdf out of the Node object and have them
// as local tables in forward/backward algorithms

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
  Hypergraph() : last_node_evaluated() {}
  ~Hypergraph();
  // construct a graph
  VariableIndex add_input(real** ps);
  VariableIndex add_input(real s, real** ps = 0);
  VariableIndex add_input(const Matrix& m, Matrix** pm = 0);
  VariableIndex add_input(const Dim& d, Matrix** pm = 0);
  VariableIndex add_parameter(Parameters* p);
  // use pindex to point to a memory location where the index will live
  // that the caller owns
  VariableIndex add_lookup(LookupParameters* p, unsigned* pindex);
  VariableIndex add_lookup(LookupParameters* p, unsigned index);
  template <class Function> inline VariableIndex add_function(const std::initializer_list<VariableIndex>& arguments);
  template <class Function, typename T>
  inline VariableIndex add_function(const std::initializer_list<VariableIndex>& arguments,
                                    const T& side_information);
  template <class Function, typename T> inline VariableIndex add_function(const T& arguments);

  // perform computations
  Matrix forward();
  Matrix incremental_forward();  // if you want to add nodes and evaluate just the new parts
  void backward();

  // debugging
  void PrintGraphviz() const;

  // data
  std::vector<Node*> nodes;  // **stored in topological order**
  std::vector<Edge*> edges;  // all edges
  std::vector<ParameterEdgeBase*> parameter_edges; // edges that contain parameters that can be updated (subset of edges)
  VariableIndex last_node_evaluated; // enables forward graphs to be evaluated incrementally
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

  // computation
  // TODO remove these from here, they should be local to the forward/backward
  // algorithms
  Matrix f;               // f(x_1 , ... , x_n)
  Matrix dEdf;            // dE/df
};

inline void swap(Node& n1, Node& n2) {
  using std::swap;
  n1.f.swap(n2.f);
  n1.dEdf.swap(n2.dEdf);
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

  // computation
  virtual Matrix forward(const std::vector<const Matrix*>& xs) const = 0;
  // computes the derivative of E with respect to the ith argument to f, that is, xs[i]
  virtual Matrix backward(const std::vector<const Matrix*>& xs,
                          const Matrix& fx,
                          const Matrix& dEdf,
                          unsigned i) const = 0;
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
template <class Function, typename T>
inline VariableIndex Hypergraph::add_function(const std::initializer_list<VariableIndex>& arguments,
                                              const T& side_information) {
  VariableIndex new_node_index(nodes.size());
  unsigned new_edge_index = edges.size();
  nodes.push_back(new Node(new_edge_index, new_node_index));
  Edge* new_edge = new Function(side_information);
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

} // namespace cnn

#endif
