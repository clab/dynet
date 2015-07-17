#ifndef CNN_GRU_H_
#define CNN_GRU_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"

namespace cnn {

class Model;

struct GRUBuilder : public RNNBuilder {
  GRUBuilder() = default;
  explicit GRUBuilder(unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      Model* model);
  std::vector<Expression> final_h() const { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const { return final_h(); }
  void rewind_one_step() { h.pop_back(); }
  Expression back() const { return h.back().back(); }

 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(const Expression& x) override;

  // first index is layer, then ...
  std::vector<std::vector<Parameters*>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<Expression>> h;

  // initial values of h at each layer
  // - default to zero matrix input
  std::vector<Expression> h0;

  unsigned hidden_dim;
  unsigned layers;
  std::vector<float> zeros;

  // the state machine ensures that the caller is behaving
  RNNStateMachine sm;
};

} // namespace cnn

#endif
