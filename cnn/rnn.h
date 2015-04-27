#ifndef CNN_RNN_H_
#define CNN_RNN_H_

#include "cnn/cnn.h"
#include "cnn/edges.h"

namespace cnn {

class Model;

struct RNNBuilder {
  RNNBuilder() {}
  explicit RNNBuilder(unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      Model* model);

  // call this to reset the builder when you are going to create
  // a new computation graph
  void new_graph();

  // call this before add_input
  void add_parameter_edges(Hypergraph* hg);

  // Reset for new sequence on hypergraph hg with shared parameters
  // call this before add_input and after add_parameter_edges, or
  // when starting a new sequence on the same hypergraph.
  // h_0 is used to initialize hidden layers at timestep 0 to given values
  void start_new_sequence(Hypergraph* hg, std::vector<VariableIndex> h_0={});

  // add another timestep by reading in the variable x
  // return the hidden representation of the deepest layer
  VariableIndex add_input(VariableIndex x, Hypergraph* hg);

  // rewind the last timestep - this DOES NOT remove the variables
  // from the computation graph, it just means the next time step will
  // see a different previous state. You can remind as many times as
  // you want.
  void rewind_one_step() {
    h.pop_back();
  }

  // returns node (index) of most recent output
  VariableIndex back() const { return h.back().back(); }

  // access hidden state contents
  const std::vector<VariableIndex>& final_h() const { return h.back(); }

  // check to make sure parameters have been added before adding input
  unsigned builder_state;

  // first index is layer, then x2h h2h hb
  std::vector<std::vector<Parameters*>> params;

  // first index is layer, then x2h h2h hb
  std::vector<std::vector<VariableIndex>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<VariableIndex>> h;

  // initial value of h
  // defaults to zero matrix input
  std::vector<VariableIndex> h0;

  Hypergraph* hg;

  unsigned hidden_dim;
  unsigned layers;
  std::vector<float> zeros;
};

} // namespace cnn

#endif
