#include "cnn/cnn.h"
#include "cnn/cnn-edges.h"

using namespace std;

namespace cnn {

Edge::~Edge() {}

Hypergraph::~Hypergraph() {
  for (auto e : edges) delete e;
  for (auto n : nodes) delete n;
}

unsigned Hypergraph::add_parameter(const Dim& d, const string& name) {
  unsigned new_node_index = nodes.size();
  nodes.push_back(new Node(edges.size(), name));
  edges.push_back(new ParameterEdge(d));
  edges.back()->head_node = new_node_index;
  return new_node_index;
}

unsigned Hypergraph::add_input(const Dim& d, const string& name) {
  unsigned new_node_index = nodes.size();
  nodes.push_back(new Node(edges.size(), name));
  edges.push_back(new InputEdge(d));
  edges.back()->head_node = new_node_index;
  return new_node_index;
}

Matrix Hypergraph::forward() {
  for (auto node : nodes) { // nodes are stored in topological order
    const Edge& in_edge = *edges[node->in_edge];
    vector<const Matrix*> xs(in_edge.arity());
    unsigned ti = 0;
    for (unsigned tail_node_index : in_edge.tail) {
      xs[ti] = &nodes[tail_node_index]->f;
      ++ti;
    }
    node->f = in_edge.forward(xs);
    node->dEdf = Zero(Dim(node->f.rows(), node->f.cols()));
  }
  return nodes.back()->f;
}

void Hypergraph::backward() {
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
      Node& tail_node = *nodes[in_edge.tail[ti]];
      tail_node.dEdf += in_edge.backward(xs, node.f, node.dEdf, ti);
    }
  }
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

