#include "cnn/cnn.h"
#include "cnn/exec.h"
#include "cnn/nodes.h"
#include "cnn/param-nodes.h"
#include "cnn/aligned-mem-pool.h"

using namespace std;

namespace cnn {

int n_hgs = 0;

Node::~Node() {}
bool Node::has_parameters() const {
  return false;
}

ComputationGraph::ComputationGraph() : last_node_evaluated(),
  ee(new SimpleExecutionEngine(*this)) {
  ++n_hgs;
  if (n_hgs > 1) {
    // TODO handle memory better
    cerr << "Memory allocator assumes only a single hypergraph at a time.\n";
    abort();
  }
}

ComputationGraph::~ComputationGraph() {
  for (auto n : nodes) delete n;
  --n_hgs;
}

VariableIndex ComputationGraph::add_input(real s) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new ScalarInputNode(s));
  return new_node_index;
}

VariableIndex ComputationGraph::add_input(const real* ps) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new ScalarInputNode(ps));
  return new_node_index;
}

VariableIndex ComputationGraph::add_input(const Dim& d, const vector<float>* pm) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new InputNode(d, pm));
  return new_node_index;
}

VariableIndex ComputationGraph::add_parameter(Parameters* p) {
  VariableIndex new_node_index(nodes.size());
  ParameterNode* new_node = new ParameterNode(p);
  nodes.push_back(new_node);
  parameter_nodes.push_back(new_node_index);
  return new_node_index;
}

VariableIndex ComputationGraph::add_lookup(LookupParameters* p, const unsigned* pindex) {
  VariableIndex new_node_index(nodes.size());
  LookupNode* new_node = new LookupNode(p, pindex);
  nodes.push_back(new_node);
  parameter_nodes.push_back(new_node_index);
  return new_node_index;
}

VariableIndex ComputationGraph::add_lookup(LookupParameters* p, unsigned index) {
  VariableIndex new_node_index(nodes.size());
  LookupNode* new_node = new LookupNode(p, index);
  nodes.push_back(new_node);
  parameter_nodes.push_back(new_node_index);
  return new_node_index;
}

VariableIndex ComputationGraph::add_const_lookup(LookupParameters* p, unsigned* pindex) {
  VariableIndex new_node_index(nodes.size());
  LookupNode* new_node = new LookupNode(p, pindex);
  // get rid of this in favor of using parameter_nodes to see the needs_derivative
  // expression
  new_node->has_optimizable_parameters = false;
  nodes.push_back(new_node);
  return new_node_index;
}

VariableIndex ComputationGraph::add_const_lookup(LookupParameters* p, unsigned index) {
  VariableIndex new_node_index(nodes.size());
  LookupNode* new_node = new LookupNode(p, index);
  new_node->has_optimizable_parameters = false;
  nodes.push_back(new_node);
  return new_node_index;
}

const Tensor& ComputationGraph::incremental_forward() { return ee->incremental_forward(); }
const Tensor& ComputationGraph::forward() { return ee->forward(); }
void ComputationGraph::backward() { ee->backward(); }

void ComputationGraph::PrintGraphviz() const {
  cerr << "digraph G {\n  rankdir=LR;\n  nodesep=.05;\n";
  unsigned nc = 0;
  for (auto node : nodes) {
    vector<string> var_names;
    for (auto arg : node->args)
      var_names.push_back(string("v")); // TODO
    cerr << "  N" << nc << " [label=\"v" << nc << " = "
         << node->as_string(var_names) << "\"];\n";
    for (auto arg : node->args)
      cerr << "  N" << ((unsigned)arg) << " -> N" << nc << ";\n";
    ++nc;
  }
  cerr << "}\n";
}

}  // namespace cnn

