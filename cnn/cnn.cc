#include "cnn/cnn.h"
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

ComputationGraph::ComputationGraph() : last_node_evaluated() {
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
  parameter_nodes.push_back(new_node);
  return new_node_index;
}

VariableIndex ComputationGraph::add_lookup(LookupParameters* p, const unsigned* pindex) {
  VariableIndex new_node_index(nodes.size());
  LookupNode* new_node = new LookupNode(p, pindex);
  nodes.push_back(new_node);
  parameter_nodes.push_back(new_node);
  return new_node_index;
}

VariableIndex ComputationGraph::add_lookup(LookupParameters* p, unsigned index) {
  VariableIndex new_node_index(nodes.size());
  LookupNode* new_node = new LookupNode(p, index);
  nodes.push_back(new_node);
  parameter_nodes.push_back(new_node);
  return new_node_index;
}

VariableIndex ComputationGraph::add_const_lookup(LookupParameters* p, unsigned* pindex) {
  VariableIndex new_node_index(nodes.size());
  LookupNode* new_node = new LookupNode(p, pindex);
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

const Tensor& ComputationGraph::incremental_forward() {
  // free any old memory if this is a new HG
  if (last_node_evaluated == 0) {
    fxs->free();
    dEdfs->zero_and_free();
  }

  assert(nodes.size() > 0);
  if (nodes.size() - last_node_evaluated == 0) {
    return nodes.back()->f;
  }
  vector<Dim> xds;
  for (unsigned i = last_node_evaluated; i < nodes.size(); ++i) {
    Node* node = nodes[i];
    xds.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      xds[ai] = nodes[arg]->dim;
      ++ai;
    }
    // TODO remove dim_forward and replace it with the Node constructors
    node->dim = node->dim_forward(xds);
    node->f.d = node->dim;
    node->f.v = static_cast<float*>(fxs->allocate(node->dim.size() * sizeof(float)));
    node->dEdf.d = node->dim;
    node->dEdf.v = static_cast<float*>(dEdfs->allocate(node->dim.size() * sizeof(float)));
    assert(node->f.v);
    assert(node->dEdf.v);
  }

  //vector<string> dummy(5, "x");
  vector<const Tensor*> xs;
  while (last_node_evaluated < nodes.size()) {
    Node* node = nodes[last_node_evaluated];
    xs.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      xs[ai] = &nodes[arg]->f;
      ++ai;
    }

    // we pass in node->f rather than expecting forward to know where it lives
    // because we may end up batching up operations in a later version
    node->forward(xs, node->f);
    ++last_node_evaluated;
  }
  return nodes.back()->f;
}

const Tensor& ComputationGraph::forward() {
  last_node_evaluated = 0;
  return incremental_forward();
}

void ComputationGraph::backward() {
  if (nodes.back()->dim.size() != 1) {
    cerr << "backward() called on non-scalar node.\n";
    abort();
  }
  // here we find constants to avoid doing extra work
  vector<bool> needs_derivative(nodes.size(), false);
  for (unsigned ni = 0; ni < nodes.size(); ++ni) {
    const Node& node = *nodes[ni];
    bool is_variable = node.has_parameters();
    for (auto arg : node.args)
      is_variable |= needs_derivative[arg];
    needs_derivative[ni] = is_variable;
  }

  // initialize dE/dE = 1
  nodes.back()->dEdf.v[0] = 1;

  // loop in reverse topological order
  vector<const Tensor*> xs;
  for (int i = nodes.size() - 1; i >= 0; --i) {
    const Node& node = *nodes[i];
    unsigned ai = 0;
    xs.resize(node.arity());
    for (VariableIndex arg : node.args) {
      xs[ai] = &nodes[arg]->f;
      ++ai;
    }
    for (unsigned ai = 0; ai < node.args.size(); ++ai) {
      if (needs_derivative[node.args[ai]]) {
        Node& arg_node = *nodes[node.args[ai]];
        node.backward(xs, node.f, node.dEdf, ai, arg_node.dEdf);
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
  for (auto pnode : parameter_nodes)
    pnode->accumulate_grad(pnode->dEdf);
}

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

