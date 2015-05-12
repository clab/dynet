#include "cnn/cnn.h"
#include "cnn/edges.h"
#include "cnn/param-edges.h"
#include "cnn/aligned-mem-pool.h"

using namespace std;

namespace cnn {

Edge::~Edge() {}

bool Edge::has_parameters() const { return false; }

Hypergraph::Hypergraph() { }

Hypergraph::~Hypergraph() {
  for (auto n : nodes) delete n;
  for (auto e : edges) delete e;
}

VariableIndex Hypergraph::add_input(real s) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  ScalarInputEdge* e = new ScalarInputEdge(s);
  edges.push_back(e);
  edges.back()->head_node = new_node_index;
  return new_node_index;
}

VariableIndex Hypergraph::add_input(const real* ps) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  ScalarInputEdge* e = new ScalarInputEdge(ps);
  edges.push_back(e);
  edges.back()->head_node = new_node_index;
  return new_node_index;
}

VariableIndex Hypergraph::add_input(const Dim& d, const vector<float>* pm) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  InputEdge* e = new InputEdge(d, pm);
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

VariableIndex Hypergraph::add_lookup(LookupParameters* p, const unsigned* pindex) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  LookupEdge* new_edge = new LookupEdge(p, pindex);
  edges.push_back(new_edge);
  parameter_edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  return new_node_index;
}

VariableIndex Hypergraph::add_lookup(LookupParameters* p, unsigned index) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  LookupEdge* new_edge = new LookupEdge(p, index);
  edges.push_back(new_edge);
  parameter_edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  return new_node_index;
}

VariableIndex Hypergraph::add_const_lookup(LookupParameters* p, unsigned* pindex) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  LookupEdge* new_edge = new LookupEdge(p, pindex);
  new_edge->has_optimizable_parameters = false;
  edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  return new_node_index;
}

VariableIndex Hypergraph::add_const_lookup(LookupParameters* p, unsigned index) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Node(edges.size(), new_node_index));
  LookupEdge* new_edge = new LookupEdge(p, index);
  new_edge->has_optimizable_parameters = false;
  edges.push_back(new_edge);
  new_edge->head_node = new_node_index;
  return new_node_index;
}

const Tensor& Run::incremental_forward() {
  // free any old memory if this is a new HG
  if (last_node_evaluated == 0) {
    fxs->free();
    dEdfs->zero_and_free();
  }

  assert(hg->nodes.size() > 0);
  if (hg->nodes.size() - last_node_evaluated == 0) {
    return runnodes[hg->nodes.size()-1].f;
  }

  if (runnodes.size() < hg->nodes.size())
    runnodes.resize(hg->nodes.size());

  vector<Dim> xds;
  for (unsigned i = last_node_evaluated; i < hg->nodes.size(); ++i) {
    Node* node = hg->nodes[i];
    RunNode& runnode = runnodes[i];
    const Edge& in_edge = *hg->edges[node->in_edge];
    xds.resize(in_edge.arity());
    unsigned ti = 0;
    for (VariableIndex tail_node_index : in_edge.tail) {
      xds[ti] = runnodes[tail_node_index].dim;
      ++ti;
    }
    runnode.dim = in_edge.dim_forward(xds);
    runnode.f.d = runnode.dim;
    runnode.f.v = static_cast<float*>(fxs->allocate(runnode.dim.size() * sizeof(float)));
    runnode.dEdf.d = runnode.dim;
    runnode.dEdf.v = static_cast<float*>(dEdfs->allocate(runnode.dim.size() * sizeof(float)));
    assert(runnode.f.v);
    assert(runnode.dEdf.v);
  }

  //vector<string> dummy(5, "x");
  vector<const Tensor*> xs;
  while (last_node_evaluated < hg->nodes.size()) {
    Node* node = hg->nodes[last_node_evaluated];
    RunNode& runnode = runnodes[last_node_evaluated];
    const Edge& in_edge = *hg->edges[node->in_edge];
    xs.resize(in_edge.arity());
    unsigned ti = 0;
    for (VariableIndex tail_node_index : in_edge.tail) {
      xs[ti] = &runnodes[tail_node_index].f;
      ++ti;
    }
    in_edge.forward(xs, runnode.f);
    ++last_node_evaluated;
  }
  return runnodes.back().f;
}

const Tensor& Run::forward() {
  last_node_evaluated = 0;
  return incremental_forward();
}

void Run::backward() {
  // here we find constants to avoid doing extra work
  vector<bool> needs_derivative(hg->nodes.size(), false);
  for (unsigned ni = 0; ni < hg->nodes.size(); ++ni) {
    const Node& node = *hg->nodes[ni];
    const Edge& in_edge = *hg->edges[node.in_edge];
    bool is_variable = in_edge.has_parameters();
    for (auto tail_node : in_edge.tail)
      is_variable |= needs_derivative[tail_node];
    needs_derivative[ni] = is_variable;
  }

  // initialize dE/dE = 1
  runnodes.back().dEdf.v[0] = 1;

  // loop in reverse topological order
  vector<const Tensor*> xs;
  for (int i = hg->nodes.size() - 1; i >= 0; --i) {
    const Node& node = *hg->nodes[i];
    const Edge& in_edge = *hg->edges[node.in_edge];
    const RunNode& runnode = runnodes[i];
    unsigned ti = 0;
    xs.resize(in_edge.arity());
    for (unsigned tail_node_index : in_edge.tail) {
      xs[ti] = &runnodes[tail_node_index].f;
      ++ti;
    }
    for (unsigned ti = 0; ti < in_edge.tail.size(); ++ti) {
      if (needs_derivative[in_edge.tail[ti]]) {
        RunNode& tail_node = runnodes[in_edge.tail[ti]];
        in_edge.backward(xs, runnode.f, runnode.dEdf, ti, tail_node.dEdf);
      }
    }
  }
  //vector<string> dummy(5, "x");
  //int cc = 0; // REMOVE
  //for (auto n : nodes) { cerr << "NODE " << edges[n->in_edge]->as_string(dummy) << endl << (*n->dEdf) << endl; }
  //abort();

  // accumulate gradients into parameters
  // this is simpler than you might find in some other frameworks
  // since we assume parameters come into the graph as a "function"
  // that returns the current value of the parameters
  for (auto pedge : hg->parameter_edges)
    pedge->accumulate_grad(runnodes[pedge->head_node].dEdf);
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

