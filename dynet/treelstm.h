#pragma once
#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"

namespace dynet {

struct TreeLSTMBuilder : public RNNBuilder {
public:
  virtual Expression back() const override;
  virtual std::vector<Expression> final_h() const override;
  virtual std::vector<Expression> final_s() const override;
  virtual unsigned num_h0_components() const override;
  virtual void copy(const RNNBuilder & params) override;
  virtual Expression add_input(int id, std::vector<int> children, const Expression& x) = 0;
  std::vector<Expression> get_h(RNNPointer i) const override { throw std::runtime_error("get_h() not a valid function for TreeLSTMBuilder"); }
  std::vector<Expression> get_s(RNNPointer i) const override { throw std::runtime_error("get_s() not a valid function for TreeLSTMBuilder"); }
  Expression set_s_impl(int prev, const std::vector<Expression>& s_new) override { throw std::runtime_error("set_s_impl() not a valid function for TreeLSTMBuilder"); }
 protected:
  virtual void new_graph_impl(ComputationGraph& cg, bool update) override = 0;
  virtual void start_new_sequence_impl(const std::vector<Expression>& h0) override = 0;
  virtual Expression add_input_impl(int prev, const Expression& x) override;
};

struct NaryTreeLSTMBuilder : public TreeLSTMBuilder {
  NaryTreeLSTMBuilder() = default;
  explicit NaryTreeLSTMBuilder(unsigned N, //Max branching factor
                       unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model);

  Expression add_input(int id, std::vector<int> children, const Expression& x) override;
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

struct UnidirectionalTreeLSTMBuilder : public TreeLSTMBuilder {
  UnidirectionalTreeLSTMBuilder() = default;
  explicit UnidirectionalTreeLSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model);

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

struct BidirectionalTreeLSTMBuilder : public TreeLSTMBuilder {
  BidirectionalTreeLSTMBuilder() = default;
  explicit BidirectionalTreeLSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model);

  Expression add_input(int id, std::vector<int> children, const Expression& x) override;
  ParameterCollection & get_parameter_collection() override {
    return local_model;
  }
 protected:
  void new_graph_impl(ComputationGraph& cg, bool update) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression set_h_impl(int prev, const std::vector<Expression>& h_new) override;

 public:
  LSTMBuilder fwd_node_builder;
  LSTMBuilder rev_node_builder;
  std::vector<Expression> h;
  ParameterCollection local_model;
};

} // namespace dynet
