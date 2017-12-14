/**
 * \file treelstm.h
 * \brief Helper structures to build tree-structured recurrent units
 */

#pragma once
#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"

namespace dynet {

/**
 * \ingroup rnnbuilders
 * \brief TreeLSTMBuilder is the base class for tree structured lstm builders.
 */
struct TreeLSTMBuilder : public RNNBuilder {
public:
  virtual Expression back() const override;
  virtual std::vector<Expression> final_h() const override;
  virtual std::vector<Expression> final_s() const override;
  virtual unsigned num_h0_components() const override;
  virtual void copy(const RNNBuilder & params) override;

  /**
   *
   * \brief add input with given children at position id \details if
   * you did not call `set_num_elems` before, each successive id must
   * be the previous id plus one and the children must all be smaller
   * than id.
   * If you used `set_num_elems`, id must be smaller than the number
   * of elements and the children must have been already provided.
   *
   * \param id index where `x` should be stored
   * \param children indices of the children for x
   */
  virtual Expression add_input(int id, std::vector<int> children, const Expression& x) = 0;

  /**
   *
   * \brief Set the number of nodes in your tree in advance
   * \details By default, input to a TreeLSTMBuilder needs to be in
   * ascending order, i.e. when sequentializing the nodes, all leaves
   * have to be first.  If you know the number of elements beforehand,
   * you can call this method to then place your nodes at arbitrary
   * indices, e.g. because you already have a sequentialization that
   * does not conform to the leaves-first requirement.
   *
   * \param num desired size
   */
  virtual void set_num_elements(int num) = 0;

  // methods declared for sequence models that are not applicable to tree structured lstms
  std::vector<Expression> get_h(RNNPointer i) const override { throw std::runtime_error("get_h() not a valid function for TreeLSTMBuilder"); }
  std::vector<Expression> get_s(RNNPointer i) const override { throw std::runtime_error("get_s() not a valid function for TreeLSTMBuilder"); }
  Expression set_s_impl(int prev, const std::vector<Expression>& s_new) override { throw std::runtime_error("set_s_impl() not a valid function for TreeLSTMBuilder"); }
  Expression set_h_impl(int prev, const std::vector<Expression>& h_new) override { throw std::runtime_error("set_h() not a valid function for TreeLSTMBuilder"); }

 protected:
  virtual void new_graph_impl(ComputationGraph& cg, bool update) override = 0;
  virtual void start_new_sequence_impl(const std::vector<Expression>& h0) override = 0;
  virtual Expression add_input_impl(int prev, const Expression& x) override;
};

  
/**
 * \ingroup rnnbuilders
 * \brief Builds N-ary trees with a fixed upper bound of children.
 * \detail See "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
 * by Tai, Nary, and Manning (2015), section 3.2, for details on this model.
 * http://arxiv.org/pdf/1503.00075v3.pdf
 */
struct NaryTreeLSTMBuilder : public TreeLSTMBuilder {
  NaryTreeLSTMBuilder() = default;
  explicit NaryTreeLSTMBuilder(unsigned N, //Max branching factor
                       unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model);

  Expression add_input(int id, std::vector<int> children, const Expression& x) override;
  void set_num_elements(int length) override;
  void copy(const RNNBuilder & params) override;
  ParameterCollection & get_parameter_collection() override;
 protected:
  void new_graph_impl(ComputationGraph& cg, bool update) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression Lookup(unsigned layer, unsigned p_type, unsigned value);

 public:
  ParameterCollection local_model;
  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;
  std::vector<std::vector<LookupParameter>> lparams;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;
  std::vector<std::vector<std::vector<Expression>>> lparam_vars;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
  unsigned N; // Max branching factor
private:
  ComputationGraph* cg;
};

/**
 * \ingroup rnnbuilders
 * \brief Builds a tree-LSTM which is recursively defined by a
 * unidirectional LSTM over the node and its children representations
 */
struct UnidirectionalTreeLSTMBuilder : public TreeLSTMBuilder {
  UnidirectionalTreeLSTMBuilder() = default;
  explicit UnidirectionalTreeLSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model);

  void set_num_elements(int length) override;
  Expression add_input(int id, std::vector<int> children, const Expression& x) override;
  ParameterCollection & get_parameter_collection() override { return node_builder.get_parameter_collection(); }
 protected:
  void new_graph_impl(ComputationGraph& cg, bool update) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;

 public:
  ParameterCollection local_model;
  LSTMBuilder node_builder;
  std::vector<Expression> h;
};

/**
 * \ingroup rnnbuilders
 * \brief Builds a tree-LSTM which is recursively defined by a
 * Bidirectional LSTM over the node and its children representations
 */
struct BidirectionalTreeLSTMBuilder : public TreeLSTMBuilder {
  BidirectionalTreeLSTMBuilder() = default;
  explicit BidirectionalTreeLSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model);

  void set_num_elements(int length) override;
  Expression add_input(int id, std::vector<int> children, const Expression& x) override;
  ParameterCollection & get_parameter_collection() override {
    return local_model;
  }
 protected:
  void new_graph_impl(ComputationGraph& cg, bool update) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;

 public:
  LSTMBuilder fwd_node_builder;
  LSTMBuilder rev_node_builder;
  std::vector<Expression> h;
  ParameterCollection local_model;
};

} // namespace dynet
