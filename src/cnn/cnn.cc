#include "cnn/cnn.h"
#include "cnn/edges.h"
#include "cnn/param-edges.h"

using namespace std;

namespace cnn {

Edge::~Edge() {}

bool Edge::has_parameters() const { return false; }

Hypergraph::~Hypergraph() {
  for (auto n : nodes) delete n;
  for (auto e : edges) delete e;
  // don't delete parameter_edges since they're a subset of edges
}

VariableIndex Hypergraph::add_input(ConstParameters* p) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  InputEdge* e = new InputEdge(p);
  edges.push_back(e);
  edges.back()->head_node = new_node_index;
  return new_node_index;
}

VariableIndex Hypergraph::add_parameter(Parameters* p) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  ParameterEdge* new_edge = new ParameterEdge(p);
  edges.push_back(new_edge);
  parameter_edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  return new_node_index;
}

VariableIndex Hypergraph::add_lookup(LookupParameters* p, unsigned** ppindex) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  LookupEdge* new_edge = new LookupEdge(p);
  *ppindex = &new_edge->index;
  edges.push_back(new_edge);
  parameter_edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  return new_node_index;
}

VariableIndex Hypergraph::add_lookup(LookupParameters* p, unsigned index) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  LookupEdge* new_edge = new LookupEdge(p);
  new_edge->index = index;
  edges.push_back(new_edge);
  parameter_edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  return new_node_index;
}

Matrix Hypergraph::incremental_forward() {
  while (last_node_evaluated < nodes.size()) {
    Node* node = nodes[last_node_evaluated];
    const Edge& in_edge = *edges[node->in_edge];
    vector<const Matrix*> xs(in_edge.arity());
    unsigned ti = 0;
    for (VariableIndex tail_node_index : in_edge.tail) {
      xs[ti] = &nodes[tail_node_index]->f;
      ++ti;
    }
    node->f = in_edge.forward(xs);
    node->dEdf = Zero(Dim(node->f.rows(), node->f.cols()));
    ++last_node_evaluated;
  }
  return nodes.back()->f;
}

Matrix Hypergraph::forward() {
  last_node_evaluated = 0;
  return incremental_forward();
}

void Hypergraph::backward() {
  // here we find constants to avoid doing extra work
  vector<bool> needs_derivative(nodes.size(), false);
  for (unsigned ni = 0; ni < nodes.size(); ++ni) {
    const Node& node = *nodes[ni];
    const Edge& in_edge = *edges[node.in_edge];
    bool is_variable = in_edge.has_parameters();
    for (auto tail_node : in_edge.tail)
      is_variable |= needs_derivative[tail_node];
    needs_derivative[ni] = is_variable;
  }

  // initialize dE/dE = 1
  nodes.back()->dEdf = Matrix(1,1);
  nodes.back()->dEdf(0,0) = 1;

  // loop in reverse topological order
  for (int i = nodes.size() - 1; i >= 0; --i) {
    const Node& node = *nodes[i];
    const Edge& in_edge = *edges[node.in_edge];
    vector<const Matrix*> xs(in_edge.arity());
    unsigned ti = 0;
    for (unsigned tail_node_index : in_edge.tail) {
      xs[ti] = &nodes[tail_node_index]->f;
      ++ti;
    }
    for (unsigned ti = 0; ti < in_edge.tail.size(); ++ti) {
      if (needs_derivative[in_edge.tail[ti]]) {
        Node& tail_node = *nodes[in_edge.tail[ti]];
        tail_node.dEdf += in_edge.backward(xs, node.f, node.dEdf, ti);
      }
    }
  }

  // accumulate gradients into parameters
  for (auto pedge : parameter_edges)
    pedge->accumulate_grad(nodes[pedge->head_node]->dEdf);
}

void Hypergraph::PrintGraphviz() const {
  cerr << "digraph G {\n  rankdir=LR;\n  nodesep=.05;\n";
  unsigned nc = 0;
  for (auto node : nodes) {
    vector<string> var_names;
    const Edge* in_edge = edges[node->in_edge];
    for (auto tail_node : in_edge->tail)
      var_names.push_back(nodes[tail_node]->variable_name());
    cerr << "  N" << nc << " [label=\"" << node->variable_name() << " = "
         << in_edge->as_string(var_names) << "\"];\n";
    ++nc;
  }
  for (auto edge : edges)
    for (auto ni : edge->tail)
      cerr << "  N" << ni << " -> N" << edge->head_node << ";\n";
  cerr << "}\n";
}

}  // namespace cnn

