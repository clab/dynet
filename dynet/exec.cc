#include "dynet/exec.h"

#include <unordered_map>

#include "dynet/param-nodes.h"
#include "dynet/globals.h"

using namespace std;

namespace dynet {

ExecutionEngine::~ExecutionEngine() {}

void SimpleExecutionEngine::invalidate() {
  num_nodes_evaluated = 0;
  backward_computed = 0;
}

void SimpleExecutionEngine::invalidate(unsigned i) {
  num_nodes_evaluated = i;
}

const Tensor& SimpleExecutionEngine::forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return forward(node_max_index);
}

const Tensor& SimpleExecutionEngine::forward(VariableIndex i) {
  invalidate();
  return incremental_forward(i);
}

const Tensor& SimpleExecutionEngine::get_value(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in SimpleExecutionEngine::get_value()");
  if (i >= num_nodes_evaluated) {
    incremental_forward();
  }
  return nfxs[i];
}

const Tensor& SimpleExecutionEngine::get_gradient(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in SimpleExecutionEngine::get_value()");
  if (i >= backward_computed) {
    DYNET_RUNTIME_ERR("Requested gradient for node " << i << ", but backward pass was computed from node " << backward_computed);
  }
  return ndEdfs[i];
}

const Tensor& SimpleExecutionEngine::incremental_forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return incremental_forward(node_max_index);
}

const Tensor& SimpleExecutionEngine::incremental_forward(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in SimpleExecutionEngine::incremental_forward()");

  // free any old memory if this is a new CG
  if (num_nodes_evaluated == 0)
    for(Device* dev : dynet::devices)
      dev->pools[(int)DeviceMempool::FXS]->free();

  if (i >= num_nodes_evaluated) {
    nfxs.resize(i + 1);

    //vector<string> dummy(5, "x");
    vector<const Tensor*> xs(16);
    for (; num_nodes_evaluated <= i; ++num_nodes_evaluated) {
      const Node* node = cg.nodes[num_nodes_evaluated];
      xs.resize(node->arity());
      unsigned ai = 0;
      for (VariableIndex arg : node->args) {
        xs[ai] = &nfxs[arg];
        ++ai;
      }
      nfxs[num_nodes_evaluated].d = node->dim;
      // Get the device
      DYNET_ASSERT(node->device != nullptr, "Attempt to access null device in SimpleExecutionEngine::incremental_forward");
      nfxs[num_nodes_evaluated].device = node->device;
      nfxs[num_nodes_evaluated].mem_pool = DeviceMempool::FXS;
      // Get the memory
      nfxs[num_nodes_evaluated].v = static_cast<float*>(nfxs[num_nodes_evaluated].device->pools[(int)DeviceMempool::FXS]->allocate(node->dim.size() * sizeof(float)));
      if (nfxs[num_nodes_evaluated].v == nullptr)
        DYNET_RUNTIME_ERR("Ran out of memory when executing node " << num_nodes_evaluated);
      void* aux_mem = nullptr;
      size_t aux_size = node->aux_storage_size();
      if (aux_size) {
        aux_mem = nfxs[num_nodes_evaluated].device->pools[(int)DeviceMempool::FXS]->allocate(aux_size);
        if (!aux_mem)
          DYNET_RUNTIME_ERR("Ran out of auxiliary memory when executing node " << num_nodes_evaluated);
      }
      node->aux_mem = aux_mem;

      node->forward(xs, nfxs[num_nodes_evaluated]);
    }
  }
  return nfxs[i];
}

void SimpleExecutionEngine::backward(bool full) {
  DYNET_ASSERT(nfxs.size() >= cg.nodes.size(), "Mismatched array sizes in SimpleExecutionEngine::backward");
  backward((VariableIndex)(cg.nodes.size()-1),full);
}

// TODO what is happening with parameter nodes if from_where > param_node_id ?
void SimpleExecutionEngine::backward(VariableIndex from_where, bool full) {
  if(!(from_where < nfxs.size()))
    incremental_forward(from_where);
  if (nfxs[from_where].d.size() != 1)
    DYNET_INVALID_ARG("backward() can only be called on scalar nodes, but node " << from_where << " has dimension: " << nfxs[from_where].d);

  const unsigned num_nodes = from_where+1;
  ndEdfs.resize(num_nodes);
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->free();
  for (unsigned i = 0; i < num_nodes; ++i) {
    const auto dim = nfxs[i].d;
    ndEdfs[i].d = dim;
    ndEdfs[i].device = nfxs[i].device;
    ndEdfs[i].mem_pool = DeviceMempool::DEDFS;
    ndEdfs[i].v = static_cast<float*>(ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(dim.size() * sizeof(float)));
    if (!ndEdfs[i].v)
      DYNET_RUNTIME_ERR("out of memory while attempting to allocate space for derivatives of node " << i);
  }
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->zero_allocated_memory();
  // initialize dE/dE = 1
  ndEdfs.back().v = kSCALAR_ONE;

  // here we find constant paths to avoid doing extra work
  // by default, a node is constant unless
  //   1) it is a parameter node
  //   2) it depends on a non-constant node
  // (thus, functions of constants and inputs end up being
  //  false in this computation)
  vector<bool> needs_derivative(num_nodes, full);
  if (!full) {
    for (auto i : cg.parameter_nodes)
      needs_derivative[i] = true;

    for (unsigned ni = 0; ni < num_nodes; ++ni) {
      bool nd = needs_derivative[ni];
      for (auto arg : cg.nodes[ni]->args)
        nd |= needs_derivative[arg];
      needs_derivative[ni] = nd;
    }
  }

  // loop in reverse topological order
  // consider only nodes that participate in the computation.
  vector<bool> in_computation(num_nodes, false);
  in_computation[num_nodes - 1] = true;
  vector<const Tensor*> xs;
  for (int i = num_nodes - 1; i >= 0; --i) {
    if (!in_computation[i]) continue;
    const Node* node = cg.nodes[i];
    xs.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      in_computation[arg] = true;
      xs[ai] = &nfxs[arg];
      ++ai;
    }
    ai = 0;
    for (VariableIndex arg : node->args) {
      if (needs_derivative[arg]) {
        node->backward(xs, nfxs[i], ndEdfs[i], ai, ndEdfs[arg]);
      }
      ++ai;
    }
  }

  // accumulate gradients into parameters
  // this is simpler than you might find in some other frameworks
  // since we assume parameters come into the graph as a "function"
  // that returns the current value of the parameters
  for (VariableIndex i : cg.parameter_nodes)
    static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
  backward_computed = from_where;
}

void BatchedExecutionEngine::invalidate() {
  num_nodes_evaluated = 0;
  backward_computed = 0;
}

void BatchedExecutionEngine::invalidate(unsigned i) {
  num_nodes_evaluated = i;
}

const Tensor& BatchedExecutionEngine::forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return forward(node_max_index);
}

const Tensor& BatchedExecutionEngine::forward(VariableIndex i) {
  invalidate();
  return incremental_forward(i);
}

const Tensor& BatchedExecutionEngine::get_value(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in BatchedExecutionEngine::get_value()");
  if (i >= num_nodes_evaluated) {
    incremental_forward();
  }
  return nfxs[i];
}

const Tensor& BatchedExecutionEngine::get_gradient(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in BatchedExecutionEngine::get_value()");
  if (i >= backward_computed) {
    DYNET_RUNTIME_ERR("Requested gradient for node " << i << ", but backward pass was computed from node " << backward_computed);
  }
  return ndEdfs[i];
}

const Tensor& BatchedExecutionEngine::incremental_forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return incremental_forward(node_max_index);
}

const Tensor& BatchedExecutionEngine::incremental_forward(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in BatchedExecutionEngine::incremental_forward()");

  // free any old memory if this is a new CG
  if (num_nodes_evaluated == 0)
    for(Device* dev : dynet::devices)
      dev->pools[(int)DeviceMempool::FXS]->free();

  if (i >= num_nodes_evaluated) {
    nfxs.resize(i + 1);

    // 1) Calculate the batching profiles for every node
    unordered_map<string, int> prof2id(i);        // Batching profile to ID
    prof2id[""] = 0;
    vector<int> node2id(i + 1, -1);  // Node to ID
    vector<int> node2left(i + 1, 0); // Node to # of predecessors left
    vector<vector<int> > node2successors(i + 1); // Node to # of predecessors left
    // Average ID of batched items, a heuristic for which to run first
    vector<float> prof2avg(i - num_nodes_evaluated + 2, 0.f), prof2cnt(i - num_nodes_evaluated + 2, 0.f);
    // The active items that cannot or can be batched
    vector<int> active_unbatched;
    vector<vector<int> > active_batched;
    int id = 0;
    for (VariableIndex j = num_nodes_evaluated; j < i; ++j) {
      const Node* node = cg.nodes[j];
      // Count the remaining input nodes to be computed for each node
      for (VariableIndex arg : node->args) {
        if(arg >= num_nodes_evaluated) {
          node2left[j]++;
          node2successors[arg].push_back(j);
        }
      }
      // Get the node profile ID
      string prof = node->autobatch_profile();
      // If batchable, collect statistics
      if(prof != "") {
        auto it = prof2id.find(prof);
        if(it == prof2id.end()) {
          id = prof2id[prof] = node2id[j] = prof2id.size();
          active_batched.resize(prof2id.size());
        } else {
          id = node2id[j] = it->second;
        }
        prof2avg[id] += j;
        prof2cnt[id]++;
        if(node2left[j] == 0)
          active_batched[id].push_back(j);
      } else if(node2left[j] == 0) {
        active_unbatched.push_back(j);
      }
    }
    for(int j = 1; j < prof2id.size(); ++j)
      prof2avg[j] /= prof2cnt[j];

    // 2) Travel through and do active nodes
    vector<const Tensor*> xs(16);
    while(num_nodes_evaluated != i + 1) {
      
      // First find the best node to execute next in order of priority
      // 1. Nodes that don't support batching
      // 2. Nodes that do support batching, but only one appropriate node is available
      // 3. Nodes that support batching and have multiple inputs available
      // In cases of 2. and 3., use a heuristic of picking the node with the lowest average
      // ID of nodes of that profile
      int curr_node = -1, curr_prof = -1;
      if(active_unbatched.size()) {
        curr_node = active_unbatched.back(); active_unbatched.pop_back();
      } else {
        float best_avg = 1e10;
        for(size_t i = 1; i < active_batched.size(); ++i) {
          if(active_batched[i].size() == 1) {
            if(curr_node == -1 || best_avg > prof2avg[i]) {
              curr_node = active_batched[i][0];
              best_avg = prof2avg[i];
            }
          } else if(active_batched[i].size() > 1) {
            if(best_avg > prof2avg[i]) {
              curr_prof = i;
              best_avg = prof2avg[i];
            } 
          }
        }
      }

      // 2.a) If we have a single current node, then we execute it
      if(curr_node != -1) {
        const Node* node = cg.nodes[curr_node];
        DYNET_ASSERT(node->device != nullptr, "Attempt to access null device in BatchedExecutionEngine::incremental_forward");
        xs.resize(node->arity());
        unsigned ai = 0;
        for (VariableIndex arg : node->args) {
          xs[ai] = &nfxs[arg];
          ++ai;
        }
        nfxs[curr_node].d = node->dim;
        nfxs[curr_node].device = node->device;
        nfxs[curr_node].mem_pool = DeviceMempool::FXS;
        nfxs[curr_node].v = static_cast<float*>(nfxs[curr_node].device->pools[(int)DeviceMempool::FXS]->allocate(node->dim.size() * sizeof(float)));
        if (nfxs[curr_node].v == nullptr)
          DYNET_RUNTIME_ERR("Ran out of memory when executing node " << curr_node);
        size_t aux_size = node->aux_storage_size();
        if (aux_size) {
          node->aux_mem = nfxs[curr_node].device->pools[(int)DeviceMempool::FXS]->allocate(aux_size);
          if (!node->aux_mem)
            DYNET_RUNTIME_ERR("Ran out of auxiliary memory when executing node " << curr_node);
        }
        node->forward(xs, nfxs[curr_node]);
        // Decrement the counts of the predecessors and add them to the active queue as appropriate
        for(auto next_node : node2successors[curr_node]) {
          if(--node2left[next_node] == 0) {
            if(node2id[next_node] == 0)
              active_unbatched.push_back(next_node);
            else
              active_batched[node2id[next_node]].push_back(next_node);
          }
        }
      // 2.b) If we have a batch of current nodes, execute them together
      } else {
        DYNET_ASSERT(curr_prof != -1, "Must have either a single node or a batch to execute");
        DYNET_RUNTIME_ERR("Executing multiple batched nodes not implemented yet.");
      }

    }
  }
  return nfxs[i];
}

void BatchedExecutionEngine::backward(bool full) {
  DYNET_ASSERT(nfxs.size() >= cg.nodes.size(), "Mismatched array sizes in BatchedExecutionEngine::backward");
  backward((VariableIndex)(cg.nodes.size()-1),full);
}

// TODO what is happening with parameter nodes if from_where > param_node_id ?
void BatchedExecutionEngine::backward(VariableIndex from_where, bool full) {
  if(!(from_where < nfxs.size()))
    incremental_forward(from_where);
  if (nfxs[from_where].d.size() != 1)
    DYNET_INVALID_ARG("backward() can only be called on scalar nodes, but node " << from_where << " has dimension: " << nfxs[from_where].d);

  const unsigned num_nodes = from_where+1;
  ndEdfs.resize(num_nodes);
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->free();
  for (unsigned i = 0; i < num_nodes; ++i) {
    const auto dim = nfxs[i].d;
    ndEdfs[i].d = dim;
    ndEdfs[i].device = nfxs[i].device;
    ndEdfs[i].mem_pool = DeviceMempool::DEDFS;
    ndEdfs[i].v = static_cast<float*>(ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(dim.size() * sizeof(float)));
    if (!ndEdfs[i].v)
      DYNET_RUNTIME_ERR("out of memory while attempting to allocate space for derivatives of node " << i);
  }
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->zero_allocated_memory();
  // initialize dE/dE = 1
  ndEdfs.back().v = kSCALAR_ONE;

  // here we find constant paths to avoid doing extra work
  // by default, a node is constant unless
  //   1) it is a parameter node
  //   2) it depends on a non-constant node
  // (thus, functions of constants and inputs end up being
  //  false in this computation)
  vector<bool> needs_derivative(num_nodes, full);
  if (!full) {
    for (auto i : cg.parameter_nodes)
      needs_derivative[i] = true;

    for (unsigned ni = 0; ni < num_nodes; ++ni) {
      bool nd = needs_derivative[ni];
      for (auto arg : cg.nodes[ni]->args)
        nd |= needs_derivative[arg];
      needs_derivative[ni] = nd;
    }
  }

  // loop in reverse topological order
  // consider only nodes that participate in the computation.
  vector<bool> in_computation(num_nodes, false);
  in_computation[num_nodes - 1] = true;
  vector<const Tensor*> xs;
  for (int i = num_nodes - 1; i >= 0; --i) {
    if (!in_computation[i]) continue;
    const Node* node = cg.nodes[i];
    xs.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      in_computation[arg] = true;
      xs[ai] = &nfxs[arg];
      ++ai;
    }
    ai = 0;
    for (VariableIndex arg : node->args) {
      if (needs_derivative[arg]) {
        node->backward(xs, nfxs[i], ndEdfs[i], ai, ndEdfs[arg]);
      }
      ++ai;
    }
  }

  // accumulate gradients into parameters
  // this is simpler than you might find in some other frameworks
  // since we assume parameters come into the graph as a "function"
  // that returns the current value of the parameters
  for (VariableIndex i : cg.parameter_nodes)
    static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
  backward_computed = from_where;
}


} // namespace dynet
