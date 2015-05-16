#ifndef CNN_EXEC_H
#define CNN_EXEC_H

#include "cnn/cnn.h"

namespace cnn {

class ExecutionEngine {
 public:
  virtual ~ExecutionEngine();
  virtual const Tensor& forward() = 0;
  virtual const Tensor& incremental_forward() = 0;  // if you want to add nodes and evaluate just the new parts
  virtual void backward() = 0;
 protected:
  explicit ExecutionEngine(const ComputationGraph& cg) : cg(cg) {}
  const ComputationGraph& cg;
};

class SimpleExecutionEngine : public ExecutionEngine {
 public:
  explicit SimpleExecutionEngine(const ComputationGraph& cg) : ExecutionEngine(cg) {}
  const Tensor& forward() override;
  const Tensor& incremental_forward() override;  // if you want to add nodes and evaluate just the new parts
  void backward() override;
 private:
  std::vector<Tensor> nfxs;
  std::vector<Tensor> ndEdfs;
  VariableIndex last_node_evaluated;
};

} // namespace cnn

#endif
