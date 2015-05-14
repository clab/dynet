#ifndef CNN_GRU_H_
#define CNN_GRU_H_

#include "cnn/cnn.h"
#include "cnn/rnn-state-machine.h"

namespace cnn {

class Model;

struct GRUBuilder {
  GRUBuilder() {}
  explicit GRUBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model);

  // call this to reset the builder when you are working with a newly
  // created ComputationGraph object
  void new_graph(ComputationGraph* hg);

  // Start new sequence in given ComputationGraph with initial c0 and h0
  // call after add_parameter edges but before add input,
  // as well as whenever a new sequence is to be added to the graph
  void start_new_sequence(ComputationGraph* hg,
                          std::vector<VariableIndex> h_0={});

  // add another timestep by reading in the variable x
  // return the hidden representation of the deepest layer
  VariableIndex add_input(VariableIndex x, ComputationGraph* hg);

  // rewind the last timestep - this DOES NOT remove the variables
  // from the computation graph, it just means the next time step will
  // see a different previous state. You can remind as many times as
  // you want.
  void rewind_one_step() {
    h.pop_back();
  }

  // returns node index (variable) of most recent output
  VariableIndex back() const { return h.back().back(); }

  // access memory/hidden state contents
  const std::vector<VariableIndex>& final_h() const { return h.back(); }

  // first index is layer, then ...
  std::vector<std::vector<Parameters*>> params;

  // first index is layer, then ...
  std::vector<std::vector<VariableIndex>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<VariableIndex>> h;

  // initial values of h at each layer
  // - default to zero matrix input
  std::vector<VariableIndex> h0;

  unsigned hidden_dim;
  unsigned layers;
  std::vector<float> zeros;

  // the state machine ensures that the caller is behaving
  RNNStateMachine sm;
};

} // namespace cnn

#endif
