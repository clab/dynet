#include "cnn/exec.h"

#include "cnn/param-nodes.h"

using namespace std;

namespace cnn {

ExecutionEngine::~ExecutionEngine() {}

const Tensor& SimpleExecutionEngine::forward() {
  last_node_evaluated = 0;
  return incremental_forward();
}

const Tensor& SimpleExecutionEngine::incremental_forward() {
  // free any old memory if this is a new HG
  if (last_node_evaluated == 0) fxs->free();

  const unsigned node_max_index = cg.nodes.size();
  assert(node_max_index > 0);
  nfxs.resize(node_max_index);
  if (node_max_index - last_node_evaluated == 0)
    return nfxs.back();
  vector<Dim> xds;
  for (unsigned i = last_node_evaluated; i < node_max_index; ++i) {
    const Node* node = cg.nodes[i];
    xds.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      xds[ai] = nfxs[arg].d;
      ++ai;
    }
    // TODO remove dim_forward and replace it with the Node constructors
    auto dim = node->dim_forward(xds);
    nfxs[i].d = dim;
    nfxs[i].v = static_cast<float*>(fxs->allocate(dim.size() * sizeof(float)));
    assert(nfxs[i].v);
  }
  //vector<string> dummy(5, "x");
  vector<const Tensor*> xs;
  for (; last_node_evaluated < node_max_index; ++last_node_evaluated) {
    const Node* node = cg.nodes[last_node_evaluated];
    xs.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      xs[ai] = &nfxs[arg];
      ++ai;
    }
    node->forward(xs, nfxs[last_node_evaluated]);
  }
  return nfxs.back();
}

void SimpleExecutionEngine::backward() {
  if (nfxs.back().d.size() != 1) {
    cerr << "backward() called on non-scalar node.\n";
    abort();
  }
  const unsigned node_max_index = cg.nodes.size();
  ndEdfs.resize(node_max_index);
  dEdfs->free();
  for (unsigned i = 0; i < node_max_index; ++i) {
    const auto dim = nfxs[i].d;
    ndEdfs[i].d = dim;
    ndEdfs[i].v = static_cast<float*>(dEdfs->allocate(dim.size() * sizeof(float)));
    assert(ndEdfs[i].v);
  }
  dEdfs->zero_allocated_memory();
  // initialize dE/dE = 1
  ndEdfs.back().v[0] = 1;

  // here we find constants to avoid doing extra work
  const unsigned num_nodes = cg.nodes.size();
  vector<bool> needs_derivative(num_nodes, false);
  for (unsigned ni = 0; ni < num_nodes; ++ni) {
    const Node* node = cg.nodes[ni];
    bool is_variable = node->has_parameters();
    for (auto arg : node->args)
      is_variable |= needs_derivative[arg];
    needs_derivative[ni] = is_variable;
  }

  // loop in reverse topological order
  vector<const Tensor*> xs;
  for (int i = num_nodes - 1; i >= 0; --i) {
    const Node* node = cg.nodes[i];
    xs.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      xs[ai] = &nfxs[arg];
      ++ai;
    }
    ai = 0;
    for (VariableIndex arg : node->args) {
      if (needs_derivative[arg])
        node->backward(xs, nfxs[i], ndEdfs[i], ai, ndEdfs[arg]);
      ++ai;
    }
  }

  // accumulate gradients into parameters
  // this is simpler than you might find in some other frameworks
  // since we assume parameters come into the graph as a "function"
  // that returns the current value of the parameters
  for (VariableIndex i : cg.parameter_nodes)
    static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
}

} // namespace cnn
