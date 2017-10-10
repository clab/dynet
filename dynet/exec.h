#ifndef DYNET_EXEC_H
#define DYNET_EXEC_H

#include "dynet/dynet.h"

namespace dynet {

class DeviceManager;

class ExecutionEngine {
 public:
  virtual ~ExecutionEngine();
  virtual void invalidate() = 0;
  virtual void invalidate(unsigned) = 0;
  virtual const Tensor& forward() = 0;
  virtual const Tensor& forward(VariableIndex i) = 0;
  // forward on multiple nodes
  virtual std::vector<const Tensor*> forward(
      const std::vector<VariableIndex>& node_list);
  // if you want to add nodes and evaluate just the new parts
  virtual const Tensor& incremental_forward() = 0;
  virtual const Tensor& incremental_forward(VariableIndex i) = 0;
  virtual const Tensor& get_value(VariableIndex i) = 0;
  virtual const Tensor& get_gradient(VariableIndex i) = 0;
  virtual void backward(bool full = false) = 0;
  virtual void backward(VariableIndex i, bool full = false) = 0;
 protected:
  explicit ExecutionEngine(const ComputationGraph& cg);
  DeviceManager* const device_manager;
  const ComputationGraph& cg;
  VariableIndex backward_computed;
};

class SimpleExecutionEngine : public ExecutionEngine {
 public:
  explicit SimpleExecutionEngine(const ComputationGraph& cg) :
    ExecutionEngine(cg), num_nodes_evaluated(0) {}
  void invalidate() override;
  void invalidate(unsigned i) override;
  const Tensor& forward() override;
  const Tensor& forward(VariableIndex i) override;
  const Tensor& incremental_forward() override;
  const Tensor& incremental_forward(VariableIndex i) override;
  const Tensor& get_value(VariableIndex i) override;
  const Tensor& get_gradient(VariableIndex i) override;
  void backward(bool full = false) override;
  void backward(VariableIndex from_where, bool full = false) override;
 private:
  std::vector<Tensor> nfxs;
  std::vector<Tensor> ndEdfs;
  VariableIndex num_nodes_evaluated;
};

struct BatchInfo {
public:
  BatchInfo() : pseudo_node(nullptr) { }
  // The forward tensor, may be null if singleton batch
  Tensor nfx;
  // The pseudo node used for calculation, also may be null if not needed
  Node* pseudo_node;
  // IDs of the batch components
  std::vector<VariableIndex> ids;
  // 0=no need to concat
  // 1=need to concat
  // 2=need to concat + already contiguous in space
  std::vector<int> concat;
  // Concatenated arguments
  std::vector<const Tensor*> arg_nfxs;
};

class BatchedExecutionEngine : public ExecutionEngine {
 public:
  explicit BatchedExecutionEngine(const ComputationGraph& cg) :
    ExecutionEngine(cg), num_nodes_evaluated(0), num_batches_evaluated(0) {}
  ~BatchedExecutionEngine() { garbage_collect(); }
  void invalidate() override;
  void invalidate(unsigned i) override;
  const Tensor& forward() override;
  const Tensor& forward(VariableIndex i) override;
  const Tensor& incremental_forward() override;
  const Tensor& incremental_forward(VariableIndex i) override;
  const Tensor& get_value(VariableIndex i) override;
  const Tensor& get_gradient(VariableIndex i) override;
  void backward(bool full = false) override;
  void backward(VariableIndex from_where, bool full = false) override;
  void garbage_collect();
 private:
  const Tensor& incremental_forward_no_update(VariableIndex upto,
                                              int autobatch_strategy);
  void combine_tensors(const std::vector<VariableIndex>& batch_ids,
                       int aid, Tensor &tout);
  void accumulate_tensors(const Tensor& tin,
                          const std::vector<VariableIndex>& batch_ids,
                          int ai);
  const Tensor& get_nfx(VariableIndex i);
  std::vector<Tensor> nfx_cache;
  std::vector<Tensor> ndEdfs;
  VariableIndex num_nodes_evaluated, num_batches_evaluated;
  // Information about the batched computation graph
  std::vector<VariableIndex> node2batch; // length: number of nodes
  std::vector<size_t> node2offset, node2size; // length: number of nodes
  std::vector<BatchInfo> batches; // length: number of batches
  SigMap sigmap;
};

} // namespace dynet

#endif
