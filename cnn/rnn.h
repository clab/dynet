#ifndef CNN_RNN_H_
#define CNN_RNN_H_

#include "cnn/cnn.h"
#include "cnn/rnn-state-machine.h"

namespace cnn {

class Model;

// interface for constructing an RNN, LSTM, GRU, etc.
struct RNNBuilder {
  virtual ~RNNBuilder();
  // call this to reset the builder when you are working with a newly
  // created ComputationGraph object
  void new_graph(ComputationGraph* cg) {
    sm.transition(RNNOp::new_graph);
    new_graph_impl(cg);
  }

  // Reset for new sequence
  // call this before add_input and after new_graph,
  // when starting a new sequence on the same hypergraph.
  // h_0 is used to initialize hidden layers at timestep 0 to given values
  void start_new_sequence(const std::vector<VariableIndex>& h_0={}) {
    sm.transition(RNNOp::start_new_sequence);
    start_new_sequence_impl(h_0);
  }

  // add another timestep by reading in the variable x
  // return the hidden representation of the deepest layer
  VariableIndex add_input(VariableIndex x, ComputationGraph* cg) {
    sm.transition(RNNOp::add_input);
    return add_input_impl(x, cg);
  }

  // rewind the last timestep - this DOES NOT remove the variables
  // from the computation graph, it just means the next time step will
  // see a different previous state. You can remind as many times as
  // you want.
  virtual void rewind_one_step() = 0;
  // returns node (index) of most recent output
  virtual VariableIndex back() const = 0;
  // access hidden state contents
  virtual std::vector<VariableIndex> final_h() const = 0;
 protected:
  virtual void new_graph_impl(ComputationGraph* cg) = 0;
  virtual void start_new_sequence_impl(const std::vector<VariableIndex>& h_0) = 0;
  virtual VariableIndex add_input_impl(VariableIndex x, ComputationGraph* cg) = 0;
 private:
  // the state machine ensures that the caller is behaving
  RNNStateMachine sm;
};

struct SimpleRNNBuilder : public RNNBuilder {
  SimpleRNNBuilder() = default;
  explicit SimpleRNNBuilder(unsigned layers,
                            unsigned input_dim,
                            unsigned hidden_dim,
                            Model* model);

 protected:
  void new_graph_impl(ComputationGraph* cg) override;
  void start_new_sequence_impl(const std::vector<VariableIndex>& h_0) override;
  VariableIndex add_input_impl(VariableIndex x, ComputationGraph* cg) override;

 public:
  void rewind_one_step() { h.pop_back(); }
  VariableIndex back() const { return h.back().back(); }
  std::vector<VariableIndex> final_h() const { return h.back(); }

 private:
  // first index is layer, then x2h h2h hb
  std::vector<std::vector<Parameters*>> params;

  // first index is layer, then x2h h2h hb
  std::vector<std::vector<VariableIndex>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<VariableIndex>> h;

  // initial value of h
  // defaults to zero matrix input
  std::vector<VariableIndex> h0;

  unsigned layers;
};

} // namespace cnn

#endif
