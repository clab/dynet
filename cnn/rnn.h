#ifndef CNN_RNN_H_
#define CNN_RNN_H_

#include "cnn/cnn.h"
#include "cnn/rnn-state-machine.h"
#include "cnn/expr.h"

using namespace cnn::expr;

namespace cnn {

class Model;

// interface for constructing an RNN, LSTM, GRU, etc.
struct RNNBuilder {
  virtual ~RNNBuilder();
  // call this to reset the builder when you are working with a newly
  // created ComputationGraph object
  void new_graph(ComputationGraph& cg) {
    sm.transition(RNNOp::new_graph);
    new_graph_impl(cg);
  }

  // Reset for new sequence
  // call this before add_input and after new_graph,
  // when starting a new sequence on the same hypergraph.
  // h_0 is used to initialize hidden layers at timestep 0 to given values
  void start_new_sequence(const std::vector<Expression>& h_0={}) {
    sm.transition(RNNOp::start_new_sequence);
    start_new_sequence_impl(h_0);
  }

  // add another timestep by reading in the variable x
  // return the hidden representation of the deepest layer
  Expression add_input(const Expression& x) {
    sm.transition(RNNOp::add_input);
    return add_input_impl(x);
  }

  // rewind the last timestep - this DOES NOT remove the variables
  // from the computation graph, it just means the next time step will
  // see a different previous state. You can remind as many times as
  // you want.
  virtual void rewind_one_step() = 0;
  // returns node (index) of most recent output
  virtual Expression back() const = 0;
  // access the final output of each hidden layer
  virtual std::vector<Expression> final_h() const = 0;
  // access the state of each hidden layer, in a format that can be used in
  // start_new_sequence
  virtual std::vector<Expression> final_s() const = 0;
 protected:
  virtual void new_graph_impl(ComputationGraph& cg) = 0;
  virtual void start_new_sequence_impl(const std::vector<Expression>& h_0) = 0;
  virtual Expression add_input_impl(const Expression& x) = 0;
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
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h_0) override;
  Expression add_input_impl(const Expression& x) override;

 public:
  void rewind_one_step() { h.pop_back(); }
  Expression back() const { return h.back().back(); }
  std::vector<Expression> final_h() const { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const { return final_h(); }

 private:
  // first index is layer, then x2h h2h hb
  std::vector<std::vector<Parameters*>> params;

  // first index is layer, then x2h h2h hb
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<Expression>> h;

  // initial value of h
  // defaults to zero matrix input
  std::vector<Expression> h0;

  unsigned layers;
};

} // namespace cnn

#endif
