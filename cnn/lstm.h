#ifndef CNN_LSTM_H_
#define CNN_LSTM_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"

namespace cnn {

class Model;

struct LSTMBuilder : public RNNBuilder {
  LSTMBuilder() = default;
  explicit LSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model);

  void rewind_one_step() {
    h.pop_back();
    c.pop_back();
  }
  VariableIndex back() const { return h.back().back(); }
  std::vector<VariableIndex> final_h() const { return h.back(); }
 protected:
  void new_graph_impl(ComputationGraph* cg) override;
  void start_new_sequence_impl(const std::vector<VariableIndex>& h0) override;
  VariableIndex add_input_impl(VariableIndex x, ComputationGraph* cg) override;

 public:
  // first index is layer, then ...
  std::vector<std::vector<Parameters*>> params;

  // first index is layer, then ...
  std::vector<std::vector<VariableIndex>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<VariableIndex>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<VariableIndex> h0;
  std::vector<VariableIndex> c0;
  unsigned layers;
};

} // namespace cnn

#endif
