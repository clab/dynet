#include "dynet/exec.h"

#include <unordered_map>
#include <queue>

#include "dynet/param-nodes.h"
#include "dynet/globals.h"

using namespace std;

namespace dynet {

inline string print_vec(const std::vector<float> & vec) {
  string sep = "[";
  ostringstream oss;
  for(auto f : vec) {
    oss << sep << f; sep = ",";
  }
  oss << "]";
  return oss.str();
}

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
  num_batches_evaluated = 0;
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
  if (num_nodes_evaluated == 0) {
    for(Node* pseudo_node : batched_nodes)
      if(pseudo_node == nullptr)
        delete pseudo_node;
    for(Device* dev : dynet::devices)
      dev->pools[(int)DeviceMempool::FXS]->free();
    for(auto & kv : batched_args)
      delete kv.second;
    batched_args.clear();
  }

  if (i >= num_nodes_evaluated) {

    nfxs.resize(i + 1);
    node2batchid.resize(i + 1);

    // Create the necessary info for batching in the future
    VariableIndex node_id = num_nodes_evaluated;
    VariableIndex batch_id = num_batches_evaluated;
    batched_nfxs.resize(i - num_nodes_evaluated + num_batches_evaluated + 1);
    batched_nodes.resize(batched_nfxs.size(), nullptr);
    batched_ids.resize(batched_nfxs.size());
    batched_concats.resize(batched_nfxs.size());

    // Allocate temporary memory for bookkeeping
    unordered_map<string, int> prof2id(i);        // Batching profile to ID
    prof2id[""] = 0;
    size_t temp_data_size = (i+1)*4*sizeof(int) + (i-node_id+2)*2*sizeof(float);
    int* node2profid = (int*)malloc(temp_data_size); memset(node2profid, 0, temp_data_size);
    int* node2left = node2profid + i + 1;
    int* node2diff = node2left + i + 1;
    int* active_un_begin = node2diff + i + 1;
    int* active_un_end = active_un_begin;
    float* prof2avg = (float*)(active_un_begin + i + 1);
    float* prof2cnt = prof2avg + i - node_id + 2;

    vector<vector<VariableIndex> > node2successors(i + 1); // Node to successors
    // Average ID of batched items, a heuristic for which to run first
    // The active items that cannot or can be batched
    vector<vector<VariableIndex> > active_batched;

    // 1) Calculate the batching profiles for every node
    int id = 0;
    for (VariableIndex j = num_nodes_evaluated; j <= i; ++j) {
      const Node* node = cg.nodes[j];
      // Count the remaining input nodes to be computed for each node
      for (VariableIndex arg : node->args) {
        if(arg >= node_id) {
          node2left[j]++;
          node2successors[arg].push_back(j);
        }
      }
      // Get the node profile ID
      string prof = node->autobatch_profile(cg);
      // If batchable, collect statistics
      if(prof != "") {
        auto it = prof2id.find(prof);
        if(it == prof2id.end()) {
          id = prof2id[prof] = node2profid[j] = prof2id.size();
          active_batched.resize(prof2id.size()+1);
        } else {
          id = node2profid[j] = it->second;
        }
        prof2avg[id] += j;
        prof2cnt[id]++;
        if(node2left[j] == 0)
          active_batched[id].push_back(j);
      } else if(node2left[j] == 0) {
        *(active_un_end++) = j;
      }
    }
    for(int j = 1; j < prof2id.size(); ++j)
      prof2avg[j] /= prof2cnt[j];

    // 2) Travel through and do active nodes
    vector<const Tensor*> xs(16);
    while(node_id != i + 1) {

      // First find the best node to execute next in order of priority
      // 1. Nodes that don't support batching
      // 2. Nodes that support batching. In this case, use a heuristic
      //    of picking the node with the lowest average ID of nodes of
      //    that profile.
      int curr_node = -1, curr_prof = -1;
      if(active_un_begin != active_un_end) {
        curr_node = *(active_un_begin++);
      } else {
        float best_avg = 1e10;
        for(size_t i = 1; i < active_batched.size(); ++i) {
          if(active_batched[i].size() > 0 && best_avg > prof2avg[i]) {
            curr_prof = i;
            best_avg = prof2avg[i];
          } 
        }
        if(active_batched[curr_prof].size() == 1) {
          curr_node = active_batched[curr_prof][0];
          active_batched[curr_prof].clear();
          curr_prof = -1;
        }
      }

      // 2.a) If we have a single current node, then we execute it
      if(curr_node != -1) {
        // Set the inputs
        // cerr << "Processing single: " << curr_node << endl;
        const Node* node = cg.nodes[curr_node];
        DYNET_ASSERT(node->device != nullptr, "Attempt to access null device in BatchedExecutionEngine::incremental_forward");
        // Save the node profile
        nfxs[curr_node].d = node->dim;
        nfxs[curr_node].device = node->device;
        nfxs[curr_node].mem_pool = DeviceMempool::FXS;
        // Allocate memory
        nfxs[curr_node].v = static_cast<float*>(nfxs[curr_node].device->pools[(int)DeviceMempool::FXS]->allocate(node->dim.size() * sizeof(float)));
        if (nfxs[curr_node].v == nullptr)
          DYNET_RUNTIME_ERR("Ran out of memory when allocating for node " << curr_node);
        size_t aux_size = node->aux_storage_size();
        if (aux_size) {
          node->aux_mem = nfxs[curr_node].device->pools[(int)DeviceMempool::FXS]->allocate(aux_size);
          if (!node->aux_mem)
            DYNET_RUNTIME_ERR("Ran out of auxiliary memory when allocating for node " << curr_node);
        }
        // Decrement the counts of the predecessors and add them to the active queue as appropriate
        for(auto next_node : node2successors[curr_node]) {
          if(--node2left[next_node] == 0) {
            if(node2profid[next_node] == 0)
              *(active_un_end++) = next_node;
            else
              active_batched[node2profid[next_node]].push_back(next_node);
          }
        }
        // Create the information for the batched pseudo-graph
        batched_ids[batch_id].resize(1, (VariableIndex)curr_node);
        // Increment the counts
        node2batchid[curr_node] = batch_id;
        ++batch_id;
        ++node_id;
        // 2.b) If we have a batch of current nodes, execute them together
      } else {
        Node* node;
        DYNET_ASSERT(curr_prof != -1, "Must have either a single node or a batch to execute");
        auto & batch_ids = active_batched[curr_prof];
        batched_ids[batch_id] = active_batched[curr_prof];
        DYNET_ASSERT(batch_ids.size() > 0, "Attempting to process empty batch at " << curr_prof);
        // cerr << "Processing batched:"; for(auto bid : batch_ids) cerr << ' ' << bid; cerr << endl;
        // Set up the configuration of each component node, including pointer differential from the start of the batch
        size_t bd = 0, tot_main = 0, tot_aux = 0, my_main, my_aux;
        for(auto curr_node : batch_ids) {
          node = cg.nodes[curr_node];
          nfxs[curr_node].d = node->dim;
          bd += node->dim.bd;
          nfxs[curr_node].device = node->device;
          nfxs[curr_node].mem_pool = DeviceMempool::FXS;
          my_main = node->dim.size();
          my_aux = node->aux_storage_size();
          node2diff[curr_node] = tot_main;
          tot_main += my_main;
          node->aux_mem = (void*)my_aux; tot_aux += my_aux;
        }

        // Allocate main/auxiliary memory for the batch
        float *head_main = static_cast<float*>(node->device->pools[(int)DeviceMempool::FXS]->allocate(tot_main * sizeof(float)));
        if(head_main == nullptr) DYNET_RUNTIME_ERR("Ran out of memory when executing node " << curr_node);
        for(auto curr_node : batch_ids)
          nfxs[curr_node].v = head_main + node2diff[curr_node];
        void *head_aux = nullptr;
        if(tot_aux > 0) {
          head_aux = static_cast<void*>(node->device->pools[(int)DeviceMempool::FXS]->allocate(tot_aux));
          if(head_aux == nullptr) DYNET_RUNTIME_ERR("Ran out of memory when executing node " << curr_node);
          for(auto curr_node : batch_ids)
            cg.nodes[curr_node]->aux_mem = (void*)((ptrdiff_t)head_aux + (ptrdiff_t)cg.nodes[curr_node]->aux_mem);
        }

        // Set the size for the final output
        Tensor & nfx = batched_nfxs[batch_id];
        nfx.device = node->device;
        nfx.mem_pool = DeviceMempool::FXS;
        nfx.d = Dim({(unsigned int)tot_main});
        nfx.v = head_main;

        // Get the concatenation and pseudo-node info
        batched_concats[batch_id] = node->autobatch_concat(cg);
        batched_nodes[batch_id] = node->autobatch_pseudo_node(cg, batch_ids);
        if(batched_nodes[batch_id] != nullptr)
          batched_nodes[batch_id]->aux_mem = head_aux;

        // Decrement the counts of the predecessors and add them to the active queue as appropriate
        for(auto curr_node : batch_ids) {
          node2batchid[curr_node] = batch_id;
          for(auto next_node : node2successors[curr_node]) {
            if(--node2left[next_node] == 0) {
              if(node2profid[next_node] == 0)
                *(active_un_end++) = next_node;
              else
                active_batched[node2profid[next_node]].push_back(next_node);
            }
          }
        }

        // Clear the active things for this profile
        ++batch_id;
        node_id += active_batched[curr_prof].size();
        batch_ids.clear();
      }
    }

    // 3: do the actual execution 
    Tensor temp_nfx;
    while(num_batches_evaluated < batch_id) {
      // Read in the stuff for this batch
      if (batched_ids[num_batches_evaluated].size() == 1) { // execute a single node
        VariableIndex nid = batched_ids[num_batches_evaluated][0];
        Node* node = cg.nodes[nid];
        xs.resize(node->arity());
        unsigned ai = 0;
        for (VariableIndex arg : node->args) {
          xs[ai] = &nfxs[arg];
          ++ai;
        }
        node->forward(xs, nfxs[nid]);
        // cerr << "Single evaluation for nfxs[" << nid << "] = " << print_vec(as_vector(nfxs[nid])) << endl;
        ++num_batches_evaluated;
      } else { // execute a batch node
        vector<VariableIndex> & batch_ids = batched_ids[num_batches_evaluated];
        // cerr << "Evaluating batch " << num_batches_evaluated << ":"; for(auto bid : batch_ids) cerr << ' ' << bid; cerr << endl;
        vector<bool> & autobatch_concat = batched_concats[num_batches_evaluated];
        size_t arity = autobatch_concat.size();
        Tensor & nfx = batched_nfxs[num_batches_evaluated];
        Node* node = batched_nodes[num_batches_evaluated];
        if(node == nullptr) node = cg.nodes[batch_ids[0]];
        xs.resize(arity); 
        // Figure out whether we need to create the inputs
        for(size_t i = 0; i < arity; ++i) {
          // 1) the inputs don't need to be concatenated. Just use the tensor
          if(!autobatch_concat[i]) {
            xs[i] = &nfxs[node->args[i]];
          // 2) the inputs need to be concatenated
          } else {
            // 2.a) the inputs need to be concatenated, but are already in the right order within a contiguous block of memory
            // TODO: make this work completely
            // float* min_node = nfxs[cg.nodes[batch_ids[0]]->args[i]].v;
            // float* max_node = min_node;
            // for(auto curr_node : batch_ids) {
            //   const Tensor &t = nfxs[cg.nodes[curr_node]->args[i]];
            //   tot_arg += t.d.size();
            //   float* v = t.v;
            //   if (v < min_node) min_node = v;
            //   if (v > max_node) max_node = v + t.d.size();
            // }
            // if (min_node + tot_arg == max_node) {
            //   xs[i] = &batched_nfxs[...];
            //   autobatch_garbage[i] = false;
            // }
            // 2.b) the inputs need to be concatenated, and are not contiguous
            Tensor* my_xsi = new Tensor;
            my_xsi->device = node->device;
            my_xsi->mem_pool = DeviceMempool::FXS;
            size_t tot_arg = 0;
            for(auto curr_node : batch_ids)
              tot_arg += nfxs[cg.nodes[curr_node]->args[i]].d.size();
            my_xsi->d = Dim({(unsigned int)tot_arg});
            my_xsi->v = static_cast<float*>(node->device->pools[(int)DeviceMempool::FXS]->allocate(tot_arg * sizeof(float)));
            tot_arg = 0;
            for(auto curr_node : batch_ids) {
              temp_nfx = nfxs[cg.nodes[curr_node]->args[i]];
              temp_nfx.v = my_xsi->v + tot_arg;
              TensorTools::copy_elements(temp_nfx, nfxs[cg.nodes[curr_node]->args[i]]);
              tot_arg += temp_nfx.d.size();
            }
            xs[i] = my_xsi;
            batched_args[make_pair(num_batches_evaluated,i)] = my_xsi;
          }
        }

        node->autobatch_reshape(cg, batch_ids, autobatch_concat, xs, nfx);
        node->forward(xs, nfx);
        // cerr << "Single evaluation for batched_nfxs[" << num_batches_evaluated << "] = " << print_vec(as_vector(nfx)) << endl;
        ++num_batches_evaluated;

      }
    }

    free(node2profid);
  }

  num_nodes_evaluated = i+1;
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

  // Find the batch that the node of interest participates in
  // cerr << "Find the batch that the node of interest participates in" << endl;
  VariableIndex num_batches;
  size_t pos_in_batch = num_nodes_evaluated;
  for(num_batches = num_batches_evaluated; num_batches > 0; --num_batches) {
    const auto & batch_ids = batched_ids[num_batches-1];
    for(size_t j = 0; j < batch_ids.size(); ++j) {
      if(batch_ids[j] == from_where) {
        pos_in_batch = j;
        break;
      }
    }
    if(pos_in_batch != (size_t)num_nodes_evaluated)
      break;
  }
  DYNET_ASSERT(num_batches != (VariableIndex)0, "Couldn't find the variable in the batched IDs.");

  // Allocate the memory
  // cerr << "Allocate the memory" << endl;
  vector<Tensor> batched_ndEdfs(num_batches);
  ndEdfs.resize(nfxs.size());
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->free();
  for (unsigned i = 0; i < num_batches; ++i) {
    // cerr << "Doing batch " << i << ", batched_nfxs.size()=" << batched_nfxs.size() << ", batched_ndEdfs.size()=" << batched_ndEdfs.size() << endl;
    const auto & batch_ids = batched_ids[i];
    const auto dim = batch_ids.size() > 1 ? batched_nfxs[i].d : nfxs[batch_ids[0]].d;
    batched_ndEdfs[i].d = dim;
    batched_ndEdfs[i].device = cg.nodes[batched_ids[i][0]]->device;
    batched_ndEdfs[i].mem_pool = DeviceMempool::DEDFS;
    batched_ndEdfs[i].v = static_cast<float*>(batched_ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(dim.size() * sizeof(float)));
    if (!batched_ndEdfs[i].v)
      DYNET_RUNTIME_ERR("out of memory while attempting to allocate space for derivatives of node " << i);
    // Assign the memory within the batch
    // cerr << "Sub-allocating" << endl;
    ptrdiff_t first_offset = (ptrdiff_t)nfxs[batch_ids[0]].v;
    for(auto id : batch_ids) {
      ndEdfs[id].d = nfxs[id].d;
      ndEdfs[id].device = nfxs[id].device;
      ndEdfs[id].mem_pool = DeviceMempool::DEDFS;
      ndEdfs[id].v = batched_ndEdfs[i].v + ((ptrdiff_t)nfxs[id].v - first_offset) / sizeof(float);
      // cerr << "ndEdfs[" << id <<"].v == " << (ptrdiff_t)ndEdfs[id].v << " == " << (ptrdiff_t)batched_ndEdfs[i].v << " + " << (ptrdiff_t)nfxs[id].v << " - " << (ptrdiff_t)first_offset << " (nfxs[" << id << "].d == " << nfxs[id].d << ")" << endl;
    }
  }
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->zero_allocated_memory();
  // for(size_t i = 0; i < ndEdfs.size(); ++i) { cerr << "ndEdfs[" << i << "]: " << print_vec(as_vector(ndEdfs[i])) << ", v=" << (ptrdiff_t)ndEdfs[i].v << endl; }
  // for(size_t i = 0; i < batched_ndEdfs.size(); ++i) { cerr << "batched_ndEdfs[" << i << "]: " << print_vec(as_vector(batched_ndEdfs[i])) << ", v=" << (ptrdiff_t)batched_ndEdfs[i].v << endl; }

  // initialize dE/dE = 1
  // cerr << "Initialize dE/dE = 1" << endl;
  size_t final_size = batched_ndEdfs.back().d.size();
  if(final_size == 1) {
    TensorTools::set_element(batched_ndEdfs.back(), 0, 1);
  } else {
    vector<float> vals(final_size, 0.0f);
    vals[pos_in_batch] = 1.0f;
    TensorTools::set_elements(batched_ndEdfs.back(), vals);
  }
  // for(size_t i = 0; i < ndEdfs.size(); ++i) { cerr << "ndEdfs[" << i << "]: " << print_vec(as_vector(ndEdfs[i])) << ", v=" << (ptrdiff_t)ndEdfs[i].v << endl; }
  // for(size_t i = 0; i < batched_ndEdfs.size(); ++i) { cerr << "batched_ndEdfs[" << i << "]: " << print_vec(as_vector(batched_ndEdfs[i])) << ", v=" << (ptrdiff_t)batched_ndEdfs[i].v << endl; }

  // here we find constant paths to avoid doing extra work
  // by default, a node is constant unless
  //   1) it is a parameter node
  //   2) it depends on a non-constant node
  // (thus, functions of constants and inputs end up being
  //  false in this computation)
  // cerr << "Finding constant paths" << endl;
  vector<bool> needs_derivative(num_batches, full);
  if (!full) {
    for (auto i : cg.parameter_nodes)
      needs_derivative[node2batchid[i]] = true;  
    for (unsigned bi = 0; bi < num_batches; ++bi) {
      bool nd = needs_derivative[bi];
      for (auto ni : batched_ids[bi])
        for (auto arg : cg.nodes[ni]->args)
          nd |= needs_derivative[node2batchid[arg]];
      needs_derivative[bi] = nd;
    }
  }

  // loop in reverse topological order
  // consider only batches that participate in the computation.
  // cerr << "Executing nodes" << endl;
  vector<bool> in_computation(num_batches, false);
  in_computation.back() = true;
  vector<const Tensor*> xs;
  for (int i = num_batches - 1; i >= 0; --i) {
    // cerr << "At batch " << i << endl;
    if (!in_computation[i]) continue;
    if (batched_ids[i].size() == 1) { // execute a single node
      VariableIndex nid = batched_ids[i][0];
      // cerr << "Executing single backward for " << nid << endl;
      const Node* node = cg.nodes[nid];
      xs.resize(node->arity());
      unsigned ai = 0;
      for (VariableIndex arg : node->args) {
        in_computation[node2batchid[arg]] = true;
        xs[ai] = &nfxs[arg];
        ++ai;
      }
      ai = 0;
      for (VariableIndex arg : node->args) {
        if (needs_derivative[node2batchid[arg]])
          node->backward(xs, nfxs[nid], ndEdfs[nid], ai, ndEdfs[arg]);
        // cerr << "node_>backward(xs, nfxs[" << nid << "], ndEdfs[" << nid << "], " << ai << ", ndEdfs[" << arg << "])" << endl;
        // cerr << "ndEdfs[" << nid << "] = " << ndEdfs[nid].tvec() << endl;
        // cerr << "ndEdfs[" << arg << "] = " << ndEdfs[arg].tvec() << endl;
        ++ai;
      }
    } else { // execute a batch node
      vector<VariableIndex> & batch_ids = batched_ids[i];
      // cerr << "Executing batched backward for"; for(auto bid : batch_ids) cerr << ' ' << bid; cerr << endl;
      vector<bool> & autobatch_concat = batched_concats[i];
      size_t arity = autobatch_concat.size();
      Node* node = batched_nodes[i];
      if(node == nullptr) node = cg.nodes[batch_ids[0]];
      xs.resize(arity); 
      size_t ai = 0;
      for (VariableIndex arg : node->args) {
        in_computation[node2batchid[arg]] = true;
        if(!autobatch_concat[ai]) {
          xs[ai] = &nfxs[arg];
        } else {
          xs[ai] = batched_args[make_pair((VariableIndex)i, ai)];
        }
        ++ai;
      }
      ai = 0;
      for (VariableIndex arg : node->args) {
        // cerr << "Doing argument " << ai << " for batch" << endl;
        if (!autobatch_concat[ai]) {
          if (needs_derivative[node2batchid[arg]]) {
            node->backward(xs, batched_nfxs[i], batched_ndEdfs[i], ai, batched_ndEdfs[node2batchid[arg]]);
            // cerr << "node->backward(xs (";
            // for(auto x : xs) cerr << x->d << "[" << print_vec(as_vector(*x)) << "] ";
            // cerr << "), batched_nfxs[" << i << "] (" << batched_nfxs[i].d << "[" << print_vec(as_vector(batched_nfxs[i])) << "]), batched_ndEdfs[" << i << "] ("<<batched_ndEdfs[i].d<<"[" << print_vec(as_vector(batched_ndEdfs[i])) << "]), " << ai << ", batched_ndEdfs[" << node2batchid[arg] << "] ("<<batched_ndEdfs[node2batchid[arg]].d<<"[" << print_vec(as_vector(batched_ndEdfs[node2batchid[arg]])) << "])" << endl;
          }
        } else {
          bool nd = false;
          for(auto nid : batched_ids[i])
            if((bool)(nd = needs_derivative[node2batchid[cg.nodes[nid]->args[ai]]]))
              break;
          if (nd) {
            size_t used = node->device->pools[(int)DeviceMempool::DEDFS]->used();
            Tensor my_ndEdf = *xs[ai], temp_ndEdf;
            my_ndEdf.v = static_cast<float*>(batched_ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(my_ndEdf.d.size() * sizeof(float)));
            // cerr << "my_ndEdf.v == " << (size_t)my_ndEdf.v << endl;
            my_ndEdf.mem_pool = DeviceMempool::DEDFS;
            TensorTools::zero(my_ndEdf);
            node->backward(xs, batched_nfxs[i], batched_ndEdfs[i], ai, my_ndEdf);
            // cerr << "node->backward(xs (";
            // for(auto x : xs)
            //   cerr << x->d << "[" << print_vec(as_vector(*x)) << "] ";
            // cerr << "), batched_nfxs[" << i << "] (" << batched_nfxs[i].d << "[" << print_vec(as_vector(batched_nfxs[i])) << "]), batched_ndEdfs[" << i << "] ("<<batched_ndEdfs[i].d<<"[" << print_vec(as_vector(batched_ndEdfs[i])) << "]), " << ai << ", my_ndEdf ("<<my_ndEdf.d<<"[" << print_vec(as_vector(my_ndEdf)) << "])" << endl;
            size_t tot_arg = 0;
            for(auto curr_node : batch_ids) {
              temp_ndEdf = ndEdfs[cg.nodes[curr_node]->args[ai]];
              temp_ndEdf.v = my_ndEdf.v + tot_arg;
              // cerr << "copying into ndEdfs["<<cg.nodes[curr_node]->args[ai]<<"].v == " << (size_t)ndEdfs[cg.nodes[curr_node]->args[ai]].v << " from " << (size_t)temp_ndEdf.v << endl;
              TensorTools::copy_elements(ndEdfs[cg.nodes[curr_node]->args[ai]], temp_ndEdf);
              tot_arg += temp_ndEdf.d.size();
            }
            node->device->pools[(int)DeviceMempool::DEDFS]->set_used(used);
          }
        }
        // cerr << "Resulting gradients from argument " << ai << endl;
        // for(size_t i = 0; i < ndEdfs.size(); ++i) { cerr << "ndEdfs[" << i << "]: " << print_vec(as_vector(ndEdfs[i])) << ", v=" << (ptrdiff_t)ndEdfs[i].v << endl; }
        // for(size_t i = 0; i < batched_ndEdfs.size(); ++i) { cerr << "batched_ndEdfs[" << i << "]: " << print_vec(as_vector(batched_ndEdfs[i])) << ", v=" << (ptrdiff_t)batched_ndEdfs[i].v << endl; }
        ++ai;
      }
    }
  }

  // accumulate gradients into parameters
  // this is simpler than you might find in some other frameworks
  // since we assume parameters come into the graph as a "function"
  // that returns the current value of the parameters
  // TODO: Can this be batched? Maybe not with the current assumptions, but
  //       it would be nice to have.
  for (VariableIndex i : cg.parameter_nodes)
    static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
  backward_computed = from_where;
}


} // namespace dynet
