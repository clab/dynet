/**
 * \file rnn.h
 * \defgroup rnnbuilders rnnbuilders
 * \brief Helper structures to build recurrent units
 *
 * \details TODO: Create documentation and explain rnns, etc...
 */

#ifndef DYNET_RNN_H_
#define DYNET_RNN_H_

#include "dynet/dynet.h"
#include "dynet/rnn-state-machine.h"
#include "dynet/expr.h"

namespace dynet {

class ParameterCollection;

typedef int RNNPointer;
inline void swap(RNNPointer& i1, RNNPointer& i2) {
  RNNPointer t = i1; i1 = i2; i2 = t;
}

/**
 * \ingroup rnnbuilders
 * \brief interface for constructing an RNN, LSTM, GRU, etc.
 * \details [long description]
 */
struct RNNBuilder {
  /**
   *
   * \brief Default constructor
   */
  RNNBuilder() : cur(-1) {}
  virtual ~RNNBuilder();

  /**
   *
   * \brief Get pointer to the current state
   *
   * \return Pointer to the current state
   */
  RNNPointer state() const { return cur; }

  /**
   *
   * \brief Initialize with new computation graph
   * \details call this to reset the builder when you are working with a newly
   * created ComputationGraph object
   *
   * \param cg Computation graph
   * \param update Update internal parameters while training
   */
   void new_graph(ComputationGraph& cg, bool update = true) {
    sm.transition(RNNOp::new_graph);
    new_graph_impl(cg, update);
  }

  /**
   *
   * \brief Reset for new sequence
   * \details call this before add_input and after new_graph,
   * when starting a new sequence on the same hypergraph.
   *
   * \param h_0 `h_0` is used to initialize hidden layers at timestep 0 to given values
   */
  void start_new_sequence(const std::vector<Expression>& h_0 = {}) {
    sm.transition(RNNOp::start_new_sequence);
    cur = RNNPointer(-1);
    head.clear();
    start_new_sequence_impl(h_0);
  }

  //
  /**
   *
   * \brief Explicitly set the output state of a node
   *
   * \param prev Pointer to the previous state
   * \param h_new The new hidden state
   *
   * \return The hidden representation of the deepest layer
   */
  Expression set_h(const RNNPointer& prev, const std::vector<Expression>& h_new = {}) {
    sm.transition(RNNOp::add_input);
    head.push_back(prev);
    cur = static_cast<int>(head.size()) - 1;
    return set_h_impl(prev, h_new);
  }

  //
  /**
   *
   * \brief Set the internal state of a node (for lstms/grus)
   * \details For RNNs without internal states (SimpleRNN, GRU...),
   * this has the same behaviour as `set_h`
   *
   * \param prev Pointer to the previous state
   * \param s_new The new state. Can be `{new_c[0],...,new_c[n]}`
   *  or `{new_c[0],...,new_c[n], new_h[0],...,new_h[n]}`
   *
   * \return The hidden representation of the deepest layer
   */
  Expression set_s(const RNNPointer& prev, const std::vector<Expression>& s_new = {}) {
    sm.transition(RNNOp::add_input);
    head.push_back(prev);
    cur = static_cast<int>(head.size()) - 1;
    return set_s_impl(prev, s_new);
  }

  /**
   *
   * \brief Add another timestep by reading in the variable x
   *
   * \param x Input variable
   *
   * \return The hidden representation of the deepest layer
   */
  Expression add_input(const Expression& x) {
    sm.transition(RNNOp::add_input);
    head.push_back(cur);
    int rcp = cur;
    cur = static_cast<int>(head.size()) - 1;
    return add_input_impl(rcp, x);
  }

   /**
    *
   * \brief Add another timestep, with arbitrary recurrent connection.
   * \details This allows you to define a recurrent connection to `prev`
   * rather than to `head[cur]`.
   * This can be used to construct trees, implement beam search, etc.
   *
   * \param prev Pointer to the previous state
   * \param x Input variable
   *
   * \return The hidden representation of the deepest layer
   */
  Expression add_input(const RNNPointer& prev, const Expression& x) {
    sm.transition(RNNOp::add_input);
    head.push_back(prev);
    cur = static_cast<int>(head.size()) - 1;
    return add_input_impl(prev, x);
  }

  /**
   *
   * \brief Rewind the last timestep
   * \details - this DOES NOT remove the variables from the computation graph,
   * it just means the next time step will see a different previous state.
   * You can rewind as many times as you want.
   */
  void rewind_one_step() {
    cur = head[cur];
  }

  /**
   *
   * \brief Return the RNN state that is the parent of `p`
   * \details - This can be used in implementing complex structures
   * such as trees, etc.
   */
  RNNPointer get_head(const RNNPointer& p) {
    return head[p];
  }

  /**
   *
   * \brief Set Dropout
   *
   * \param d Dropout rate
   */
  virtual void set_dropout(float d) { dropout_rate = d; }

  /**
   *
   * \brief Disable Dropout
   * \details In general, you should disable dropout at test time
   */
  virtual void disable_dropout() { dropout_rate = 0; }

  /**
   *
   * \brief Returns node (index) of most recent output
   *
   * \return Node (index) of most recent output
   */
  virtual Expression back() const = 0;
  /**
   *
   * \brief Access the final output of each hidden layer
   *
   * \return Final output of each hidden layer
   */
  virtual std::vector<Expression> final_h() const = 0;
  /**
   *
   * \brief Access the output of any hidden layer
   *
   * \param i Pointer to the step which output you want to access
   *
   * \return Output of each hidden layer at the given step
   */
  virtual std::vector<Expression> get_h(RNNPointer i) const = 0;

  /**
   *
   * \brief Access the final state of each hidden layer
   * \details This returns the state of each hidden layer,
   * in a format that can be used in start_new_sequence
   * (i.e. including any internal cell for LSTMs and the likes)
   *
   * \return vector containing, if it exists, the list of final
   * internal states, followed by the list of final outputs for
   * each layer
   */
  virtual std::vector<Expression> final_s() const = 0;
  /**
   *
   * \brief Access the state of any hidden layer
   * \details See `final_s` for details
   *
   * \param i Pointer to the step which state you want to access
   *
   * \return Internal state of each hidden layer at the given step
   */
  virtual std::vector<Expression> get_s(RNNPointer i) const = 0;

  /**
   *
   * \brief Number of components in `h_0`
   *
   * \return Number of components in `h_0`
   */
  virtual unsigned num_h0_components() const  = 0;
  /**
   *
   * \brief Copy the parameters of another builder.
   *
   * \param params RNNBuilder you want to copy parameters from.
   */
  virtual void copy(const RNNBuilder & params) = 0;

  virtual ParameterCollection & get_parameter_collection() = 0;

protected:
  virtual void new_graph_impl(ComputationGraph& cg, bool update) = 0;
  virtual void start_new_sequence_impl(const std::vector<Expression>& h_0) = 0;
  virtual Expression add_input_impl(int prev, const Expression& x) = 0;
  virtual Expression set_h_impl(int prev, const std::vector<Expression>& h_new) = 0;
  virtual Expression set_s_impl(int prev, const std::vector<Expression>& c_new) = 0;
  RNNPointer cur;
  float dropout_rate;
private:
  // the state machine ensures that the caller is behaving
  RNNStateMachine sm;
  std::vector<RNNPointer> head; // head[i] returns the head position
};

/**
 * \ingroup rnnbuilders
 * \brief This provides a builder for the simplest RNN with tanh nonlinearity
 * \details The equation for this RNN is :
 * \f$h_t=\tanh(W_x x_t + W_h h_{t-1} + b)\f$
 *
 */
struct SimpleRNNBuilder : public RNNBuilder {
  SimpleRNNBuilder() = default;
  /**
   *
   * \brief Builds a simple RNN
   *
   * \param layers Number of layers
   * \param input_dim Dimension of the input
   * \param hidden_dim Hidden layer (and output) size
   * \param model ParameterCollection holding the parameters
   * \param support_lags Allow for auxiliary output?
   */
  explicit SimpleRNNBuilder(unsigned layers,
                            unsigned input_dim,
                            unsigned hidden_dim,
                            ParameterCollection& model,
                            bool support_lags = false);

protected:
  void new_graph_impl(ComputationGraph& cg, bool update) override;
  void start_new_sequence_impl(const std::vector<Expression>& h_0) override;
  Expression add_input_impl(int prev, const Expression& x) override;
  Expression set_h_impl(int prev, const std::vector<Expression>& h_new) override;
  Expression set_s_impl(int prev, const std::vector<Expression>& s_new) override {return set_h_impl(prev, s_new);}

public:
  /**
   *
   * \brief Add auxiliary output
   * \details Returns \f$h_t=\tanh(W_x x_t + W_h h_{t-1} + W_y y + b)\f$
   * where \f$y\f$ is an auxiliary output
   * TODO : clarify
   *
   * \param x Input expression
   * \param aux Auxiliary output expression
   *
   * \return The hidden representation of the deepest layer
   */
  Expression add_auxiliary_input(const Expression& x, const Expression &aux);

  Expression back() const override { return (cur == -1 ? h0.back() : h[cur].back()); }
  std::vector<Expression> final_h() const override { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const override { return final_h(); }

  std::vector<Expression> get_h(RNNPointer i) const override { return (i == -1 ? h0 : h[i]); }
  std::vector<Expression> get_s(RNNPointer i) const override { return get_h(i); }
  void copy(const RNNBuilder & params) override;


  /**
   * \brief Set the dropout rates to a unique value
   * \details This has the same effect as `set_dropout(d,d_h)` except that all the dropout rates are set to the same value.
   * \param d Dropout rate to be applied on all of \f$x,h\f$
   */
  virtual void set_dropout(float d) override;

  /**
   * \param d Dropout rate
   * \details The dropout implemented here is the variational dropout introduced in [Gal, 2016](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks)
   * More specifically, dropout masks \f$\mathbf{z_x}\sim \mathrm{Bernoulli}(1-d_x)\f$ and \f$\mathbf{z_h}\sim \mathrm{Bernoulli}(1-d_h)\f$ are sampled at the start of each sequence.
   * The dynamics of the cell are then modified to :
   *
   * \f$
   * \begin{split}
    h_t & =\tanh(W_{x}(\frac 1 {1-d}\mathbf{z_x} \circ x_t)+W_{h}(\frac 1 {1-d}\mathbf{z_h} \circ h_{t-1})+b)\\
   \end{split}
   * \f$
   *
   * For more detail as to why scaling is applied, see the "Unorthodox" section of the documentation
   * \param d Dropout rate \f$d\f$ for the input \f$x_t\f$
   */
  void set_dropout(float d, float d_h);

  virtual void disable_dropout() override;

  /**
   * \brief Set dropout masks at the beginning of a sequence for a specific bathc size
   * \details If this function is not called on batched input, the same mask will be applied across
   * all batch elements. Use this to apply different masks to each batch element
   *
   * \param batch_size Batch size
   */
  void set_dropout_masks(unsigned batch_size = 1);

  unsigned num_h0_components() const override { return layers; }

  ParameterCollection & get_parameter_collection() override;

  // first index is layer, then x2h h2h hb
  std::vector<std::vector<Parameter>> params;

  // first index is layer, then x2h h2h hb
  std::vector<std::vector<Expression>> param_vars;

private:

  ParameterCollection local_model;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h;

  // first index is layer, second specifies W_i, W_u
  std::vector<std::vector<Expression>> masks;

  // initial value of h
  // defaults to zero matrix input
  std::vector<Expression> h0;

  unsigned layers;
  unsigned input_dim_, hidden_dim_;
  bool lagging;


  float dropout_rate_h;
  bool dropout_masks_valid;

  ComputationGraph * _cg;

};

} // namespace dynet

#endif
