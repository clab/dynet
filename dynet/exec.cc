#include "dynet/exec.h"

#include <unordered_map>
#include <queue>

#include "dynet/param-nodes.h"
#include "dynet/globals.h"
#include "dynet/timing.h"

#ifdef HAVE_CUDA
#include "dynet/gpu-ops.h"
#endif

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

vector<const Tensor*> ExecutionEngine::forward(std::vector<VariableIndex> is) {
  invalidate();
  VariableIndex i=*(std::max_element(is.begin(),is.end()));
  incremental_forward(i);
  vector<const Tensor*> ret;
  for (auto i : is) {
      ret.push_back(&(get_value(i)));
  }
  return ret;
}

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
  // cerr << "SimpleExecutionEngine::incremental_forward" << endl;
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

  // for(size_t i = 0; i < ndEdfs.size(); ++i) { cerr << "ndEdfs[" << i << "]: " << print_vec(as_vector(ndEdfs[i])) << endl; }

}

// copies the list of tensors into a single contig tensor (tout).
// allocates the memory for tout.
void BatchedExecutionEngine::combine_tensors(std::vector<VariableIndex> batch_ids, int aid, Tensor &tout) {

  AlignedMemoryPool *mempool = tout.device->pools[(int)DeviceMempool::FXS];
  // determine needed memory
  VariableIndex vid;
  unsigned total_dsize = 0;
  for(auto & id : batch_ids) {
    id = cg.nodes[id]->args[aid];
    total_dsize += node2size[id];
  }
  tout.d = Dim({total_dsize});

  // allocate
  float* dest = static_cast<float*>(mempool->allocate(total_dsize * sizeof(float)));

#if HAVE_CUDA
  vector<float*> locs(batch_ids.size()*3);
  unsigned i = 0;
  unsigned max_length = 0;
  const int TRG = batch_ids.size();
  const int LEN = batch_ids.size()*2;
#endif
  tout.v = dest;
  // copy
  for (auto id : batch_ids) {
    const size_t sz = node2size[id];

    float* my_src = batches[node2batch[id]].nfx.v + node2offset[id];
#if HAVE_CUDA
    locs[i] = my_src; // src
    locs[i+TRG] = dest;
    locs[i+LEN] = (float*)sz;
    if (max_length < sz) max_length=sz;
    i++;
#else
    memcpy(dest, my_src, sz*sizeof(float));
#endif
    dest += sz; // pointer arith
  }
#if HAVE_CUDA
  size_t req_sz = batch_ids.size()*3*sizeof(float*);
  float** srcs = static_cast<float**>(mempool->allocate(req_sz));
  float** trgs = srcs + TRG;
  float** lens = srcs + LEN;
  CUDA_CHECK(cudaMemcpyAsync(srcs, &(locs)[0], locs.size()*sizeof(float**), cudaMemcpyHostToDevice));
  gpu::parallel_memcpy(batch_ids.size(), max_length, srcs, trgs, lens);
#endif
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
  return get_nfx(i);
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

void BatchedExecutionEngine::garbage_collect() {
  // free any old memory if this is a new CG
  for(auto & batch : batches) {
    if(batch.pseudo_node != nullptr)
      delete batch.pseudo_node;
    for(size_t i = 0; i < batch.arg_nfxs.size(); ++i)
      if(batch.concat[i])
        delete batch.arg_nfxs[i];
  }
  for(Device* dev : dynet::devices)
    dev->pools[(int)DeviceMempool::FXS]->free();
  batches.clear();
}

const Tensor& BatchedExecutionEngine::incremental_forward(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in BatchedExecutionEngine::incremental_forward()");
  // cerr << "BatchedExecutionEngine::incremental_forward" << endl;

  if (num_nodes_evaluated == 0)
    garbage_collect();

  if (i >= num_nodes_evaluated) {

    nfx_cache.resize(i + 1);
    node2batch.resize(i + 1);
    node2offset.resize(i + 1, 0);
    node2size.resize(i + 1, 0);

    // Create the necessary info for batching in the future
    VariableIndex node_id = num_nodes_evaluated;
    VariableIndex batch_id = num_batches_evaluated;
    batches.resize(i - num_nodes_evaluated + num_batches_evaluated + 1);

    // Allocate temporary memory for bookkeeping
    //unordered_map<string, int> prof2id(i);        // Batching profile to ID
    SigMap sigmap;   // Batching signature to id.
    size_t temp_data_size = (i+1)*4*sizeof(int) + (i-node_id+2)*2*sizeof(float);
    int* node2profid = (int*)malloc(temp_data_size); memset(node2profid, 0, temp_data_size);
    int* node2left = node2profid + i + 1;
    int* active_un_begin = node2left + i + 1;
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
      node2size[j] = node->dim.size();
      // Count the remaining input nodes to be computed for each node
      for (VariableIndex arg : node->args) {
        if(arg >= node_id) {
          node2left[j]++;
          node2successors[arg].push_back(j);
        }
      }
      // Get the node profile ID
      //string prof = node->autobatch_profile(cg);
      int sig; 
      sig = node->autobatch_sig(cg, sigmap);
      // If batchable, collect statistics
      /*
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
      */
      if (sig != 0) {
        id = node2profid[j] = sig; 
        if (active_batched.size() <= sig) active_batched.resize(sig+1);
        prof2avg[id] += j;
        prof2cnt[id]++;
        if(node2left[j] == 0)
          active_batched[id].push_back(j);
      } else if(node2left[j] == 0) {
        *(active_un_end++) = j;
      }
    }
    for(size_t j = 0; j < sigmap.size(); ++j)
      prof2avg[j] /= prof2cnt[j];

    // 2) Travel through and do active nodes
    vector<const Tensor*> xs(16), ts(16);
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
      auto & my_batch = batches[batch_id];
      auto & nfx = my_batch.nfx;
      if(curr_node != -1) {
        // Set the inputs
        // cerr << "Processing single: " << curr_node << " " << cg.nodes[curr_node]->as_dummy_string() << endl;
        const Node* node = cg.nodes[curr_node];
        DYNET_ASSERT(node->device != nullptr, "Attempt to access null device in BatchedExecutionEngine::incremental_forward");
        // Save the node profile
        nfx.d = node->dim;
        nfx.device = node->device;
        nfx.mem_pool = DeviceMempool::FXS;
        // Allocate memory
        nfx.v = static_cast<float*>(node->device->pools[(int)DeviceMempool::FXS]->allocate(node2size[curr_node] * sizeof(float)));
        if (nfx.v == nullptr)
          DYNET_RUNTIME_ERR("Ran out of memory when allocating for node " << curr_node);
        size_t aux_size = node->aux_storage_size();
        if (aux_size) {
          node->aux_mem = node->device->pools[(int)DeviceMempool::FXS]->allocate(aux_size);
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
        batches[batch_id].ids.resize(1, (VariableIndex)curr_node);
        // Increment the counts
        node2batch[curr_node] = batch_id;
        ++batch_id;
        ++node_id;
        // 2.b) If we have a batch of current nodes, execute them together
      } else {
        Node* node;
        DYNET_ASSERT(curr_prof != -1, "Must have either a single node or a batch to execute");
        auto & batch_ids = active_batched[curr_prof];
        my_batch.ids = active_batched[curr_prof];
        DYNET_ASSERT(batch_ids.size() > 0, "Attempting to process empty batch at " << curr_prof);
        // cerr << "Processing batched: " << cg.nodes[batch_ids[0]]->as_dummy_string() << ' ' ; for(auto bid : batch_ids) cerr << ' ' << bid; cerr << endl;
        // Set up the configuration of each component node, including pointer differential from the start of the batch
        size_t tot_main = 0, tot_aux = 0, my_main, my_aux;
        for(auto curr_node : batch_ids) {
          node = cg.nodes[curr_node];
          my_main = node2size[curr_node];
          my_aux = node->aux_storage_size();
          node2offset[curr_node] = tot_main;
          tot_main += my_main;
          node->aux_mem = (void*)my_aux;
          tot_aux += my_aux;
        }

        // Allocate main/auxiliary memory for the batch
        float *head_main = static_cast<float*>(node->device->pools[(int)DeviceMempool::FXS]->allocate(tot_main * sizeof(float)));
        if(head_main == nullptr) DYNET_RUNTIME_ERR("Ran out of memory when executing node " << curr_node);
        // for(auto curr_node : batch_ids)
        //   nfxs[curr_node].v = head_main + node2diff[curr_node];
        void *head_aux = nullptr;
        if(tot_aux > 0) {
          head_aux = static_cast<void*>(node->device->pools[(int)DeviceMempool::FXS]->allocate(tot_aux));
          if(head_aux == nullptr) DYNET_RUNTIME_ERR("Ran out of memory when executing node " << curr_node);
          // for(auto curr_node : batch_ids)
          //   cg.nodes[curr_node]->aux_mem = (void*)((ptrdiff_t)head_aux + (ptrdiff_t)cg.nodes[curr_node]->aux_mem);
        }

        // Set the size for the final output
        nfx.device = node->device;
        nfx.mem_pool = DeviceMempool::FXS;
        nfx.d = Dim({(unsigned int)tot_main});
        nfx.v = head_main;

        // Get the concatenation and pseudo-node info
        my_batch.concat = node->autobatch_concat(cg);
        my_batch.pseudo_node = node->autobatch_pseudo_node(cg, batch_ids);
        if(my_batch.pseudo_node != nullptr)
          my_batch.pseudo_node->aux_mem = head_aux;
        else
          cg.nodes[batch_ids[0]]->aux_mem = head_aux;


        // Decrement the counts of the predecessors and add them to the active queue as appropriate
        for(auto curr_node : batch_ids) {
          node2batch[curr_node] = batch_id;
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
      auto & my_batch = batches[num_batches_evaluated];
      if (my_batch.ids.size() == 1) { // execute a single node
        VariableIndex nid = my_batch.ids[0];
        Node* node = cg.nodes[nid];
        xs.resize(node->arity());
        unsigned ai = 0;
        for (VariableIndex arg : node->args) {
          xs[ai] = &get_nfx(arg);
          ++ai;
        }
        node->forward(xs, my_batch.nfx);
        // cerr << "Single evaluation for nfxs[" << nid << "] = " << print_vec(as_vector(my_batch.nfx)) << endl;
        ++num_batches_evaluated;
      } else { // execute a batch node
        // cerr << "Evaluating batch " << num_batches_evaluated << ":"; for(auto bid : my_batch.ids) cerr << ' ' << bid; cerr << endl;
        size_t arity = my_batch.concat.size();
        Node* node = my_batch.pseudo_node;
        if(node == nullptr) node = cg.nodes[my_batch.ids[0]];
        xs.resize(arity); 
        // Figure out whether we need to create the inputs
        my_batch.arg_nfxs.resize(arity);
        for(size_t i = 0; i < arity; ++i) {
          // 1) the inputs don't need to be concatenated. Just use the tensor
          if(!my_batch.concat[i]) {
            my_batch.arg_nfxs[i] = &batches[node2batch[node->args[i]]].nfx;
          // 2) the inputs need to be concatenated
          } else {
            // 2.a) the inputs need to be concatenated, but are already in the right order within a contiguous block of memory
            // TODO: make this work completely
            Tensor* my_xsi = new Tensor;
            my_xsi->device = node->device;
            my_xsi->mem_pool = DeviceMempool::FXS;

            // check contig memory
            float *min_node = (float*)std::numeric_limits<size_t>::max(), *max_node = (float*)0;
            unsigned int tot_arg = 0, my_arg;
            // cerr << "min_node=" << (size_t)min_node << ", max_node=" << (size_t)max_node << ", tot_arg=" << tot_arg << endl;
            for(auto curr_node : my_batch.ids) {
              VariableIndex aid = cg.nodes[curr_node]->args[i];
              float* v = batches[node2batch[aid]].nfx.v + node2offset[aid];
              my_arg = node2size[aid];
              tot_arg += my_arg;
              min_node = min(min_node, v);
              max_node = max(max_node, v + my_arg);
            }
            // cerr << "min_node=" << (size_t)min_node << ", max_node=" << (size_t)max_node << ", tot_arg=" << tot_arg << endl;
            if (min_node + tot_arg == max_node) { // if contig, use current mem for xs_i
              // cerr << "    contiguous " << my_batch.ids.size() << node->as_string() << endl;
              //xs[i] = &batched_nfxs[...];
              my_xsi->v = min_node;
              my_xsi->d = Dim({tot_arg});
            //   autobatch_garbage[i] = false;
            } else { // if non-contig, copy xs_i into new mem.
              // cerr << "non contiguous " << my_batch.ids.size() << node->as_string() << endl;
              // 2.b) the inputs need to be concatenated, and are not contiguous
              combine_tensors(my_batch.ids, i, *my_xsi);
            }
            my_batch.arg_nfxs[i] = my_xsi;
          }
        }

        node->autobatch_reshape(cg, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
        //cerr << "evluaating " << node->as_string() << endl;
        node->forward(my_batch.arg_nfxs, my_batch.nfx);
        // cerr << "Batched evaluation for batched_nfxs[" << num_batches_evaluated << "] = " << print_vec(as_vector(my_batch.nfx)) << endl;
        ++num_batches_evaluated;

      }
    }

    free(node2profid);
  }

  num_nodes_evaluated = i+1;
  return get_nfx(i);
}

void BatchedExecutionEngine::backward(bool full) {
  DYNET_ASSERT(nfxs.size() >= cg.nodes.size(), "Mismatched array sizes in BatchedExecutionEngine::backward");
  backward((VariableIndex)(cg.nodes.size()-1),full);
}

// TODO what is happening with parameter nodes if from_where > param_node_id ?
void BatchedExecutionEngine::backward(VariableIndex from_where, bool full) {

  if(!(from_where < node2batch.size()))
    incremental_forward(from_where);
  if (node2size[from_where] != 1)
    DYNET_INVALID_ARG("backward() can only be called on scalar nodes, but node " << from_where << " has dimension: " << get_nfx(from_where).d);

  // Find the batch that the node of interest participates in
  // cerr << "Find the batch that the node of interest participates in" << endl;
  VariableIndex num_batches;
  size_t pos_in_batch = num_nodes_evaluated;
  for(num_batches = num_batches_evaluated; num_batches > 0; --num_batches) {
    const auto & batch_ids = batches[num_batches-1].ids;
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
  ndEdfs.resize(node2batch.size());
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->free();
  for (unsigned i = 0; i < num_batches; ++i) {
    // cerr << "Doing batch " << i << ", batched_nfxs.size()=" << batched_nfxs.size() << ", batched_ndEdfs.size()=" << batched_ndEdfs.size() << endl;
    const auto & my_batch = batches[i];
    const auto & dim = my_batch.nfx.d;
    batched_ndEdfs[i].d = dim;
    batched_ndEdfs[i].device = cg.nodes[my_batch.ids[0]]->device;
    batched_ndEdfs[i].mem_pool = DeviceMempool::DEDFS;
    batched_ndEdfs[i].v = static_cast<float*>(batched_ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(dim.size() * sizeof(float)));
    if (!batched_ndEdfs[i].v)
      DYNET_RUNTIME_ERR("out of memory while attempting to allocate space for derivatives of node " << i);
    // Assign the memory within the batch
    // cerr << "Sub-allocating" << endl;
    for(auto id : my_batch.ids) {
      ndEdfs[id].d = cg.nodes[id]->dim;
      ndEdfs[id].device = cg.nodes[id]->device;
      ndEdfs[id].mem_pool = DeviceMempool::DEDFS;
      ndEdfs[id].v = batched_ndEdfs[i].v + node2offset[id];
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
      needs_derivative[node2batch[i]] = true;  
    for (unsigned bi = 0; bi < num_batches; ++bi) {
      bool nd = needs_derivative[bi];
      for (auto ni : batches[bi].ids)
        for (auto arg : cg.nodes[ni]->args)
          nd |= needs_derivative[node2batch[arg]];
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
    const auto & my_batch = batches[i];
    if (my_batch.ids.size() == 1) { // execute a single node
      VariableIndex nid = my_batch.ids[0];
      // cerr << "Executing single backward bid=" << i << ", nid=" << nid << endl;
      const Node* node = cg.nodes[nid];
      xs.resize(node->arity());
      unsigned ai = 0;
      for (VariableIndex arg : node->args) {
        // cerr << "in_computation_single[" << node2batch[arg] << "] = true" << endl;
        in_computation[node2batch[arg]] = true;
        xs[ai] = &get_nfx(arg);
        ++ai;
      }
      ai = 0;
      for (VariableIndex arg : node->args) {
        if (needs_derivative[node2batch[arg]]) {
          node->backward(xs, get_nfx(nid), ndEdfs[nid], ai, ndEdfs[arg]);
          // cerr << "Single evaluation for ndEdfs["<<nid<<"]("<<ai<<") -> ndEdfs[" << arg << "] = " << print_vec(as_vector(ndEdfs[arg])) << endl;
        }
        // cerr << "node_>backward(xs, nfxs[" << nid << "], ndEdfs[" << nid << "], " << ai << ", ndEdfs[" << arg << "])" << endl;
        // cerr << "ndEdfs[" << nid << "] = " << print_vec(as_vector(ndEdfs[nid])) << endl;
        // cerr << "ndEdfs[" << arg << "] = " << print_vec(as_vector(ndEdfs[arg])) << endl;
        ++ai;
      }
    } else { // execute a batch node
      // cerr << "Executing batched backward bid=" << i << ", nids="; for(auto bid : my_batch.ids) cerr << ' ' << bid; cerr << endl;
      size_t arity = my_batch.concat.size();
      Node* node = my_batch.pseudo_node;
      if(node == nullptr) node = cg.nodes[my_batch.ids[0]];
      xs.resize(arity); 
      size_t ai = 0;
      for (VariableIndex arg : node->args) {
        if(!my_batch.concat[ai]) {
          xs[ai] = &get_nfx(arg);
          in_computation[node2batch[arg]] = true;
        } else {
          xs[ai] = my_batch.arg_nfxs[ai];
          for(auto bid : my_batch.ids)
            in_computation[node2batch[cg.nodes[bid]->args[ai]]] = true;
        }
        ++ai;
      }
      ai = 0;
      for (VariableIndex arg : node->args) {
        if (!my_batch.concat[ai]) {
          if (needs_derivative[node2batch[arg]]) {
            node->backward(xs, my_batch.nfx, batched_ndEdfs[i], ai, batched_ndEdfs[node2batch[arg]]);
            // cerr << "Batched evaluation for batched_ndEdfs["<<i<<"]("<<ai<<") -> batched_ndEdfs[" << arg << "] = " << print_vec(as_vector(batched_ndEdfs[node2batch[arg]])) << endl;
          }
        } else {
          bool nd = false;
          for(auto nid : my_batch.ids)
            if((bool)(nd = needs_derivative[node2batch[cg.nodes[nid]->args[ai]]]))
              break;
          if (nd) {
            size_t used = node->device->pools[(int)DeviceMempool::DEDFS]->used();
            Tensor my_ndEdf = *xs[ai], temp_ndEdf;
            my_ndEdf.v = static_cast<float*>(batched_ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(my_ndEdf.d.size() * sizeof(float)));
            my_ndEdf.mem_pool = DeviceMempool::DEDFS;
            TensorTools::zero(my_ndEdf);
            node->backward(xs, my_batch.nfx, batched_ndEdfs[i], ai, my_ndEdf);
            // cerr << "Batched evaluation for batched_ndEdfs["<<i<<"]("<<ai<<") -> ndEdfs["; for(auto id : my_batch.ids) cerr << ' ' << id; cerr << "] = " << print_vec(as_vector(my_ndEdf)) << endl;
            size_t tot_arg = 0;
            for(auto curr_node : my_batch.ids) {
              VariableIndex my_aid = cg.nodes[curr_node]->args[ai];
              temp_ndEdf = ndEdfs[my_aid];
              temp_ndEdf.v = my_ndEdf.v + tot_arg;
              TensorTools::accumulate(ndEdfs[cg.nodes[curr_node]->args[ai]], temp_ndEdf);
              tot_arg += node2size[my_aid];
            }
            node->device->pools[(int)DeviceMempool::DEDFS]->set_used(used);
          }
        }
        // cerr << "Resulting gradients from argument " << ai << endl;
        // for(size_t i = 0; i < ndEdfs.size(); ++i) { cerr << "ndEdfs[" << i << "]: " << print_vec(as_vector(ndEdfs[i])) << endl; }
        // for(size_t i = 0; i < batched_ndEdfs.size(); ++i) { cerr << "batched_ndEdfs[" << i << "]: " << print_vec(as_vector(batched_ndEdfs[i])) << endl; }
        ++ai;
      }
    }
  }

  // for(size_t i = 0; i < ndEdfs.size(); ++i) { cerr << "ndEdfs[" << i << "]: " << print_vec(as_vector(ndEdfs[i])) << endl; }

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

const Tensor& BatchedExecutionEngine::get_nfx(VariableIndex i) {
  if(nfx_cache[i].v == nullptr) {
    const Tensor & bt = batches[node2batch[i]].nfx;
    Tensor & t = nfx_cache[i];
    t.v = bt.v + node2offset[i]; 
    t.d = cg.nodes[i]->dim;
    t.mem_pool = bt.mem_pool;
    t.device = bt.device;
  }
  return nfx_cache[i];
}

} // namespace dynet
