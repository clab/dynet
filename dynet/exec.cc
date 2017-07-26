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
    incremental_forward(i);
  }
  return nfxs[i];
}

const Tensor& SimpleExecutionEngine::get_gradient(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in SimpleExecutionEngine::get_value()");
  if (i >= backward_computed) {
    DYNET_RUNTIME_ERR("Requested gradient for node " << i << ", but backward pass was computed from node " << (backward_computed - 1));
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
    string current_node_name;
    nfxs.resize(i + 1);

    //vector<string> dummy(5, "x");
    vector<const Tensor*> xs(16);
    for (; num_nodes_evaluated <= i; ++num_nodes_evaluated) {
      const Node* node = cg.nodes[num_nodes_evaluated];
      if (autobatch_debug_flag) { 
        current_node_name = node->as_dummy_string();
        timer.start(current_node_name);
      }
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

      if (autobatch_debug_flag) { timer.stop(current_node_name); }
    }
  }

  // for(VariableIndex vi = (VariableIndex)0; vi <= i; ++vi) cerr << "nfxs[" << vi << "] == " << print_vec(as_vector(nfxs[vi])) << endl;
  return nfxs[i];
}

void SimpleExecutionEngine::backward(bool full) {
  DYNET_ASSERT(nfxs.size() >= cg.nodes.size(), "Mismatched array sizes in SimpleExecutionEngine::backward");
  backward((VariableIndex)(cg.nodes.size()-1),full);
}

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
    if(i <= from_where)
      static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
  backward_computed = from_where;
  // for(VariableIndex vi = (VariableIndex)0; vi <= backward_computed; ++vi) cerr << "ndEdfs[" << vi << "] == " << print_vec(as_vector(ndEdfs[vi])) << endl;

}

// copies the list of tensors into a single contig tensor (tout).
// allocates the memory for tout.
void BatchedExecutionEngine::combine_tensors(std::vector<VariableIndex> batch_ids, int aid, Tensor &tout) {

  AlignedMemoryPool *mempool = tout.device->pools[(int)DeviceMempool::FXS];
  // determine needed memory
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

void BatchedExecutionEngine::accumulate_tensors(const Tensor& tin, std::vector<VariableIndex> batch_ids, int ai) {

#if HAVE_CUDA
  vector<float*> locs(batch_ids.size()*3);
  unsigned i = 0;
  unsigned max_length = 0;
  const int TRG = batch_ids.size();
  const int LEN = batch_ids.size()*2;
  float* src = tin.v;
  // copy
  for (auto id : batch_ids) {
    const size_t sz = node2size[cg.nodes[id]->args[ai]];

    locs[i] = src; // src
    locs[i+TRG] = ndEdfs[cg.nodes[id]->args[ai]].v;
    locs[i+LEN] = (float*)sz;
    if (max_length < sz) max_length = sz;
    i++;
    src += sz; // pointer arith
  }
  size_t req_sz = batch_ids.size()*3*sizeof(float*);
  AlignedMemoryPool *mempool = tin.device->pools[(int)DeviceMempool::DEDFS];
  float** srcs = static_cast<float**>(mempool->allocate(req_sz));
  float** trgs = srcs + TRG;
  float** lens = srcs + LEN;
  CUDA_CHECK(cudaMemcpyAsync(srcs, &(locs)[0], locs.size()*sizeof(float**), cudaMemcpyHostToDevice));
  gpu::parallel_accumulate(batch_ids.size(), max_length, srcs, trgs, lens);
#else
  size_t tot_arg = 0;
  Tensor temp_ndEdf;
  for(auto curr_node : batch_ids) {
    VariableIndex my_aid = cg.nodes[curr_node]->args[ai];
    temp_ndEdf = ndEdfs[my_aid];
    temp_ndEdf.v = tin.v + tot_arg;
    TensorTools::accumulate(ndEdfs[cg.nodes[curr_node]->args[ai]], temp_ndEdf);
    tot_arg += node2size[my_aid];
  }
#endif

}

void BatchedExecutionEngine::invalidate() {
  num_nodes_evaluated = 0;
  num_batches_evaluated = 0;
  backward_computed = 0;
  garbage_collect();
  node2offset.clear(); node2size.clear(); node2batch.clear(); ndEdfs.clear(); nfx_cache.clear();
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
    incremental_forward(i);
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

const Tensor& BatchedExecutionEngine::incremental_forward_no_update(VariableIndex upto, int autobatch_strategy) {
  // cerr << "running graph" << endl; cg.print_graphviz();

  if (upto >= num_nodes_evaluated) {
    if (autobatch_strategy == 0) autobatch_strategy = 1;
    string current_batch_name;

    size_t uptop1 = upto + 1;

    nfx_cache.resize(uptop1);
    node2batch.resize(uptop1);
    node2offset.resize(uptop1, 0);
    node2size.resize(uptop1, 0);

    // Create the necessary info for batching in the future
    VariableIndex node_id = num_nodes_evaluated;
    VariableIndex batch_id = num_batches_evaluated;
    batches.resize(upto - num_nodes_evaluated + num_batches_evaluated + 1);

    // Allocate temporary memory for bookkeeping
    size_t temp_data_size = (uptop1)*4*sizeof(int) + (upto+2)*2*sizeof(float);
    int* node2profid = (int*)malloc(temp_data_size);
    memset(node2profid, 0, temp_data_size);
    int* node2left = node2profid + uptop1;
    int* node2depth = node2left + uptop1;
    int* active_un_begin = node2depth + uptop1;
    int* active_un_end = active_un_begin;
    float* prof2avg = (float*)(active_un_begin + uptop1);
    float* prof2cnt = prof2avg + upto - node_id + 2;

    // More intelligent batching?
    if(autobatch_strategy == 1 || autobatch_strategy == 3) {

      unordered_map<int, int> depthprofcnt(upto*3);             // Count of remaining things for this profile
      vector<VariableIndex> node2successors(uptop1,(VariableIndex)0); // Node to successors
      vector<VariableIndex> active_batched(uptop1*2,(VariableIndex)0);
      VariableIndex n2sptr, abptr, abmax = (VariableIndex)0;

      // 1) Calculate the batching profiles for every node
      int sig = 0, depth;
      for (VariableIndex j = num_nodes_evaluated; j <= upto; ++j) {
        const Node* node = cg.nodes[j];
        node2size[j] = node->dim.size();
        // Count the remaining input nodes to be computed for each node
        depth = 0;
        for (VariableIndex arg : node->args) {
          if(arg >= node_id) {
            node2left[j]++;
            n2sptr = node2successors[arg];
            node2successors.push_back(j);
            node2successors[arg] = node2successors.size();
            node2successors.push_back(n2sptr);
            depth = max(node2depth[arg]+1,depth);
          }
        }
        node2depth[j] = depth;
        // Get the node profile ID
        sig = node->autobatch_sig(cg, sigmap);
        // If batchable, collect statistics
        if (sig != 0) {
          node2profid[j] = sig; 
          if(autobatch_strategy == 3) {
            ++depthprofcnt[(depth * upto) + sig];
          }
          abmax = (VariableIndex)max((int)abmax, sig+1);
          prof2avg[sig] += depth;
          prof2cnt[sig]++;
          if(depth == 0) {
            abptr = active_batched[sig];
            ++active_batched[sig+uptop1];
            active_batched.push_back(j);
            active_batched[sig] = active_batched.size();
            active_batched.push_back(abptr);
            if(autobatch_strategy == 3)
              --depthprofcnt[sig];
          }
        } else if(node2left[j] == 0) {
          *(active_un_end++) = j;
        }
      }
      for(size_t j = 0; j < (size_t)sigmap.size(); ++j)
        prof2avg[j] /= prof2cnt[j];

      // 2) Travel through and do active nodes
      while(node_id != (VariableIndex)uptop1) {

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
          for(size_t profid = 1; profid < (size_t)abmax; ++profid) {
            const float avg = prof2avg[profid];
            if(active_batched[profid] != (VariableIndex)0 &&
               (best_avg > avg || (best_avg == avg && sigmap.sig2type(profid)<nt::COMPLEX )) && // tie-break on type, defer affine and matmul
               (autobatch_strategy == 1 || depthprofcnt[(node2depth[active_batched[active_batched[profid]-1]] * upto) + profid] == 0)) {
              curr_prof = profid;
              best_avg = avg;
            } 
          }

          abptr = active_batched[curr_prof];
          if(active_batched[abptr] == 0) {
            curr_node = active_batched[abptr-1];
            active_batched[curr_prof] = 0;
            active_batched[curr_prof + uptop1] = 0;
            curr_prof = -1;
          }
        }

        // 2.a) If we have a single current node, then we execute it
        auto & my_batch = batches[batch_id];
        if(curr_node != -1) {
          // Create the information for the batched pseudo-graph
          batches[batch_id].ids.resize(1, (VariableIndex)curr_node);
          // Increment the counts
          node2batch[curr_node] = batch_id;
          // Decrement the counts of the predecessors and add them to the active queue as appropriate
          n2sptr = node2successors[curr_node];
          while(n2sptr != (VariableIndex)0) {
            auto next_node = node2successors[n2sptr-1];
            n2sptr = node2successors[n2sptr];
            if(--node2left[next_node] == 0) {
              auto profid = node2profid[next_node];
              if(profid == 0) {
                *(active_un_end++) = next_node;
              } else {
                abptr = active_batched[profid];
                ++active_batched[profid+uptop1];
                active_batched.push_back(next_node);
                active_batched[profid] = active_batched.size();
                active_batched.push_back(abptr);
                if(autobatch_strategy == 3)
                  --depthprofcnt[(node2depth[next_node] * upto) + profid];
              }
            }
          }

          ++batch_id;
          ++node_id;
          // 2.b) If we have a batch of current nodes, execute them together
        } else {
          DYNET_ASSERT(curr_prof != -1, "Must have either a single node or a batch to execute");
          // Copy the things from the linked list to the actual batch
          abptr = active_batched[curr_prof];
          assert(abptr != (VariableIndex)0);
          my_batch.ids.resize(active_batched[curr_prof+uptop1]);
          for(auto it = my_batch.ids.rbegin(); it != my_batch.ids.rend(); ++it) {
            *it = active_batched[abptr-1];
            abptr = active_batched[abptr];
          }
          active_batched[curr_prof] = 0;
          active_batched[curr_prof+uptop1] = 0;
          auto & batch_ids = my_batch.ids;
          // Decrement the counts of the predecessors and add them to the active queue as appropriate
          size_t batch_ids_size = batch_ids.size();
          for(size_t j = 0; j < batch_ids_size; ++j) {
            VariableIndex curr_node = batch_ids[j];
            node2batch[curr_node] = batch_id;
            n2sptr = node2successors[curr_node];
            while(n2sptr != (VariableIndex)0) {
              auto next_node = node2successors[n2sptr-1];
              n2sptr = node2successors[n2sptr];
              if(--node2left[next_node] == 0) {
                auto profid = node2profid[next_node];
                if(profid == 0) {
                  *(active_un_end++) = next_node;
                } else {
                  abptr = active_batched[profid];
                  ++active_batched[profid+uptop1];
                  active_batched.push_back(next_node);
                  active_batched[profid] = active_batched.size();
                  active_batched.push_back(abptr);
                  if(autobatch_strategy == 3)
                    --depthprofcnt[(node2depth[next_node] * upto) + profid];
                }
              }
            }
          }

          // Increment
          ++batch_id;
          node_id += batch_ids_size;

        }
      }
    // depth-based batching
    } else if(autobatch_strategy == 2) {
      map<pair<int,int>, vector<VariableIndex> > depth_profile_batches;
      int sig, depth;
      Node* node;
      for (VariableIndex j = num_nodes_evaluated; j <= upto; ++j) {
        depth = 0;
        node = cg.nodes[j];
        for (auto k : node->args)
          depth = max(node2depth[k]+1,depth);
        node2depth[j] = depth;
        node2size[j] = node->dim.size();
        sig = node->autobatch_sig(cg, sigmap);
        depth_profile_batches[make_pair(depth, sig)].push_back(j); 
      }
      for(auto & batch_info : depth_profile_batches) {
        // unbatchable
        if(batch_info.first.second == 0) {
          for(auto curr_node : batch_info.second) {
            node2batch[curr_node] = batch_id;
            batches[batch_id++].ids.resize(1, curr_node);
          }
        // batchable
        } else {
          for(auto curr_node : batch_info.second)
            node2batch[curr_node] = batch_id;
          batches[batch_id++].ids = batch_info.second;
        }
      }
    }

    // 2.5 print some debug info
    if (autobatch_debug_flag) {
      cout << "Forward Call" << endl;
      for(VariableIndex bid = num_batches_evaluated; bid < batch_id; ++bid) {
        auto & batch_ids = batches[bid].ids;
        VariableIndex curr_node = batch_ids[0];
        const Node* node = cg.nodes[curr_node];
        cout << "BatchSize:" << batch_ids.size() << " " << node->as_dummy_string() << endl;
      }
    }

    // 3. Based on the batches, allocate the memory, etc
    for(VariableIndex bid = num_batches_evaluated; bid < batch_id; ++bid) {

      auto & my_batch = batches[bid];
      auto & nfx = my_batch.nfx;
      auto & batch_ids = my_batch.ids;

      if(batch_ids.size() == 1) {

        VariableIndex curr_node = batch_ids[0];
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

      } else {

        // Set up the configuration of each component node, including pointer differential from the start of the batch
        const Node* node = nullptr;
        size_t tot_main = 0, tot_aux = 0, my_main, my_aux;
        for(auto curr_node : batch_ids) {
          node = cg.nodes[curr_node];
          my_main = node2size[curr_node];
          my_aux = node->aux_storage_size();
          node2offset[curr_node] = tot_main;
          tot_main += my_main;
          node->aux_mem = (void*)tot_aux;
          tot_aux += my_aux;
        }


        // Allocate main/auxiliary memory for the batch
        float *head_main = static_cast<float*>(node->device->pools[(int)DeviceMempool::FXS]->allocate(tot_main * sizeof(float)));
        if(head_main == nullptr) DYNET_RUNTIME_ERR("Ran out of memory when executing batch " << bid);
        // for(auto curr_node : batch_ids) nfxs[curr_node].v = head_main + node2diff[curr_node];
        char *head_aux = nullptr;
        if(tot_aux > 0) {
          head_aux = static_cast<char*>(node->device->pools[(int)DeviceMempool::FXS]->allocate(tot_aux));
          if(head_aux == nullptr) DYNET_RUNTIME_ERR("Ran out of memory when executing node " << bid);
          for(auto curr_node : batch_ids)
            cg.nodes[curr_node]->aux_mem = (void*)(head_aux + (ptrdiff_t)cg.nodes[curr_node]->aux_mem);
        }

        // Get the concatenation and pseudo-node info
        my_batch.concat = node->autobatch_concat(cg);
        my_batch.pseudo_node = node->autobatch_pseudo_node(cg, batch_ids);
        if(my_batch.pseudo_node != nullptr)
          my_batch.pseudo_node->aux_mem = head_aux;
        else
          cg.nodes[batch_ids[0]]->aux_mem = head_aux;

        // Set the size for the final output
        nfx.device = node->device;
        nfx.mem_pool = DeviceMempool::FXS;
        nfx.d = Dim({(unsigned int)tot_main});
        nfx.v = head_main;

      }

    }

    // 4: do the actual execution 
    Tensor temp_nfx;
    vector<const Tensor*> xs(16), ts(16);
    while(num_batches_evaluated < batch_id) {
      // Read in the stuff for this batch
      auto & my_batch = batches[num_batches_evaluated];
      if (autobatch_debug_flag) { 
        VariableIndex nid = my_batch.ids[0];
        Node* node = cg.nodes[nid];
        current_batch_name = node->as_dummy_string();
        timer.start(current_batch_name);
      }
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
        // cerr << "unbatched forward[" << num_batches_evaluated << "] (node: " << nid << ") == " << print_vec(as_vector(my_batch.nfx)) << endl;
        ++num_batches_evaluated;
      } else { // execute a batch node
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
            auto it = my_batch.ids.begin(), itend = my_batch.ids.end();
            VariableIndex aid = cg.nodes[*(it++)]->args[i];
            float *min_node = batches[node2batch[aid]].nfx.v + node2offset[aid];
            unsigned int tot_arg = node2size[aid];
            bool contig = true;
            while(it != itend && contig) {
              aid = cg.nodes[*(it++)]->args[i];
              float* v = batches[node2batch[aid]].nfx.v + node2offset[aid];
              contig = contig && v == min_node + tot_arg;
              tot_arg += node2size[aid];
            }
            if (contig) { // if contig, use current mem for xs_i
              //xs[i] = &batched_nfxs[...];
              my_xsi->v = min_node;
              my_xsi->d = Dim({tot_arg});
              my_batch.concat[i] = 2;
            //   autobatch_garbage[i] = false;
            } else { // if non-contig, copy xs_i into new mem.
              // 2.b) the inputs need to be concatenated, and are not contiguous
              combine_tensors(my_batch.ids, i, *my_xsi);
            }
            my_batch.arg_nfxs[i] = my_xsi;
          }
        }

        node->autobatch_reshape(cg, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
        node->forward(my_batch.arg_nfxs, my_batch.nfx);
        // cerr << "batched forward[" << num_batches_evaluated << "] (nodes:"; for(auto id : my_batch.ids) cerr << ' ' << id; cerr << ") == " << print_vec(as_vector(my_batch.nfx)) << endl;
        ++num_batches_evaluated;

      }
      if (autobatch_debug_flag) { timer.stop(current_batch_name); }
    }

    free(node2profid);
  }

  // for(VariableIndex vi = (VariableIndex)0; vi <= upto; ++vi) cerr << "nfxs[" << vi << "] == " << print_vec(as_vector(get_nfx(vi))) << endl;
  return get_nfx(upto);
}

const Tensor& BatchedExecutionEngine::incremental_forward(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in BatchedExecutionEngine::incremental_forward()");

  if (num_nodes_evaluated == 0)
    garbage_collect();

  if (autobatch_flag > 99) {
    Timing timer;
    incremental_forward_no_update(i, 1);
    double best_speed = timer.stop();
    autobatch_flag = 1;
    for(size_t strat = 2; strat < 4; ++strat) {
      timer.start();
      incremental_forward_no_update(i, strat);
      double speed = timer.stop();
      if(speed < best_speed) {
        best_speed = speed;
        autobatch_flag = strat;
      }
    }
  } else {
    incremental_forward_no_update(i, autobatch_flag);
  }

  num_nodes_evaluated = max(i + 1, num_nodes_evaluated);	
  return get_nfx(i);
}

void BatchedExecutionEngine::backward(bool full) {
  DYNET_ASSERT(nfx_cache.size() >= cg.nodes.size(), "Mismatched array sizes in BatchedExecutionEngine::backward");
  backward((VariableIndex)(cg.nodes.size()-1),full);
}

void BatchedExecutionEngine::backward(VariableIndex from_where, bool full) {

  if(!(from_where < node2batch.size()))
    incremental_forward(from_where);
  if (node2size[from_where] != 1)
    DYNET_INVALID_ARG("backward() can only be called on scalar nodes, but node " << from_where << " has dimension: " << get_nfx(from_where).d);

  // Find the batch that the node of interest participates in
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
  vector<Tensor> batched_ndEdfs(num_batches);
  ndEdfs.resize(node2batch.size());
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->free();
  for (unsigned i = 0; i < num_batches; ++i) {
    const auto & my_batch = batches[i];
    const auto & dim = my_batch.nfx.d;
    batched_ndEdfs[i].d = dim;
    batched_ndEdfs[i].device = cg.nodes[my_batch.ids[0]]->device;
    batched_ndEdfs[i].mem_pool = DeviceMempool::DEDFS;
    batched_ndEdfs[i].v = static_cast<float*>(batched_ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(dim.size() * sizeof(float)));
    if (!batched_ndEdfs[i].v)
      DYNET_RUNTIME_ERR("out of memory while attempting to allocate space for derivatives of node " << i);
    // Assign the memory within the batch
    for(auto id : my_batch.ids) {
      ndEdfs[id].d = cg.nodes[id]->dim;
      ndEdfs[id].device = cg.nodes[id]->device;
      ndEdfs[id].mem_pool = DeviceMempool::DEDFS;
      ndEdfs[id].v = batched_ndEdfs[i].v + node2offset[id];
    }
  }
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->zero_allocated_memory();

  // initialize dE/dE = 1
  size_t final_size = batched_ndEdfs.back().d.size();
  if(final_size == 1) {
    TensorTools::set_element(batched_ndEdfs.back(), 0, 1);
  } else {
    vector<float> vals(final_size, 0.0f);
    vals[pos_in_batch] = 1.0f;
    TensorTools::set_elements(batched_ndEdfs.back(), vals);
  }

  // here we find constant paths to avoid doing extra work
  // by default, a node is constant unless
  //   1) it is a parameter node
  //   2) it depends on a non-constant node
  // (thus, functions of constants and inputs end up being
  //  false in this computation)
  vector<bool> needs_derivative(num_batches, full);
  if (!full) {
    for (auto i : cg.parameter_nodes)
      if(i <= from_where)
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
  vector<bool> in_computation(num_batches, false);
  in_computation.back() = true;
  vector<const Tensor*> xs;
  for (int i = num_batches - 1; i >= 0; --i) {
    if (!in_computation[i]) continue;
    const auto & my_batch = batches[i];
    if (my_batch.ids.size() == 1) { // execute a single node
      VariableIndex nid = my_batch.ids[0];
      const Node* node = cg.nodes[nid];
      xs.resize(node->arity());
      unsigned ai = 0;
      for (VariableIndex arg : node->args) {
        in_computation[node2batch[arg]] = true;
        xs[ai] = &get_nfx(arg);
        ++ai;
      }
      ai = 0;
      for (VariableIndex arg : node->args) {
        if (needs_derivative[node2batch[arg]]) {
          node->backward(xs, get_nfx(nid), ndEdfs[nid], ai, ndEdfs[arg]);
          // cerr << "unbatched backward[" << nid << "](" << ai << ")->" << arg << " == " << print_vec(as_vector(my_batch.nfx)) << endl;
        }
        ++ai;
      }
    } else { // execute a batch node
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
        // No concatenation whatsoever
        if (my_batch.concat[ai] == 0) {
          if (needs_derivative[node2batch[arg]]) {
            node->backward(xs, my_batch.nfx, batched_ndEdfs[i], ai, batched_ndEdfs[node2batch[arg]]);
            // cerr << "batched backward[" << i << "](" << ai << ")->" << node2batch[arg] << " == " << print_vec(as_vector(batched_ndEdfs[node2batch[arg]])) << endl;
          }
        // Needs concatenation
        } else {
          bool nd = false;
          for(auto nid : my_batch.ids)
            if((bool)(nd = needs_derivative[node2batch[cg.nodes[nid]->args[ai]]]))
              break;
          if (nd) {
            // Non-contiguous
            Tensor my_ndEdf = *xs[ai];
            if (my_batch.concat[ai] == 1) {
              size_t used = node->device->pools[(int)DeviceMempool::DEDFS]->used();
              my_ndEdf.v = static_cast<float*>(batched_ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(my_ndEdf.d.size() * sizeof(float)));
              my_ndEdf.mem_pool = DeviceMempool::DEDFS;
              TensorTools::zero(my_ndEdf);
              node->backward(xs, my_batch.nfx, batched_ndEdfs[i], ai, my_ndEdf);
              // cerr << "noncontig backward[" << i << "](" << ai << ")->" << node2batch[arg] << " == "; for(auto id : my_batch.ids) cerr << " ndEdfs[" << cg.nodes[id]->args[ai] << "] == " << print_vec(as_vector(ndEdfs[cg.nodes[id]->args[ai]])); cerr << " + " << print_vec(as_vector(my_ndEdf)) << " == ";
              accumulate_tensors(my_ndEdf, my_batch.ids, ai);
              // for(auto id : my_batch.ids) cerr << " ndEdfs[" << cg.nodes[id]->args[ai] << "] == " << print_vec(as_vector(ndEdfs[cg.nodes[id]->args[ai]])); cerr << endl;
              node->device->pools[(int)DeviceMempool::DEDFS]->set_used(used);
            // Contiguous
            } else {
              VariableIndex aid = cg.nodes[my_batch.ids[0]]->args[ai];
              float* v = batched_ndEdfs[node2batch[aid]].v + node2offset[aid];
              my_ndEdf.v = v;
              node->backward(xs, my_batch.nfx, batched_ndEdfs[i], ai, my_ndEdf);
              // cerr << "contig backward[" << i << "](" << ai << ")->" << node2batch[arg] << " == "; for(auto id : my_batch.ids) cerr << " ndEdfs[" << cg.nodes[id]->args[ai] << "] == " << print_vec(as_vector(ndEdfs[cg.nodes[id]->args[ai]])); cerr << endl;
            }
          }
        }
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
    if(i < (VariableIndex)ndEdfs.size() && ndEdfs[i].v != nullptr)
      static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
  backward_computed = from_where;
  // for(VariableIndex vi = (VariableIndex)0; vi <= backward_computed; ++vi) cerr << "ndEdfs[" << vi << "] == " << print_vec(as_vector(ndEdfs[vi])) << endl;

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
