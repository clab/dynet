#ifndef CNN_LSTM_H_
#define CNN_LSTM_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"

using namespace cnn::expr;

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
  Expression back() const { return h.back().back(); }
  std::vector<Expression> final_h() const { return h.back(); }
 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(const Expression& x) override;

 public:
  std::vector<AffineBuilder> i_params, o_params, c_params;

  // first index is time, second is layer 
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
};

} // namespace cnn

#endif
