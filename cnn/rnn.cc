#include "cnn/rnn.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;
using namespace cnn;

namespace cnn {

enum { X2H=0, H2H, HB, L2H };

RNNBuilder::~RNNBuilder() {}

SimpleRNNBuilder::SimpleRNNBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model, 
                       bool support_lags) : layers(layers), lagging(support_lags) {
  unsigned layer_input_dim = input_dim;
  assert(layers > 0);
  assert(input_dim > 0);
  assert(hidden_dim > 0);
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = model->add_parameters({long(hidden_dim), long(layer_input_dim)});
    Parameters* p_h2h = model->add_parameters({long(hidden_dim), long(hidden_dim)});
    Parameters* p_hb = model->add_parameters({long(hidden_dim)});
    vector<Parameters*> ps = {p_x2h, p_h2h, p_hb};
    if (lagging)
        ps.push_back(model->add_parameters({long(hidden_dim), long(hidden_dim)}));
    params.push_back(ps);
    layer_input_dim = hidden_dim;
  }
}

void SimpleRNNBuilder::new_graph_impl(ComputationGraph& cg) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = params[i][X2H];
    Parameters* p_h2h = params[i][H2H];
    Parameters* p_hb = params[i][HB];
    Expression i_x2h =  parameter(cg,p_x2h);
    Expression i_h2h =  parameter(cg,p_h2h);
    Expression i_hb =  parameter(cg,p_hb);
    vector<Expression> vars = {i_x2h, i_h2h, i_hb};

    if (lagging) {
        Parameters* p_l2h = params[i][L2H];
        Expression i_l2h =  parameter(cg,p_l2h);
        vars.push_back(i_l2h);
    }

    param_vars.push_back(vars);
  }
}

void SimpleRNNBuilder::start_new_sequence_impl(const vector<Expression>& h_0) {
  h.clear();
  h0 = h_0;
  if (h0.size()) { assert(h0.size() == layers); }
}

Expression SimpleRNNBuilder::add_input_impl(const Expression &in) {
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));

  Expression x = in;

  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];

    Expression y = vars[HB] + vars[X2H] * x;

    if (t == 0 && h0.size() > 0) 
      y = y + vars[H2H] * h0[i];
    else if (t >= 1)
      y = y + vars[H2H] * h[t-1][i];

    x = h[t][i] = tanh(y);
  }
  return h[t].back();
}

Expression SimpleRNNBuilder::add_auxiliary_input(const Expression &in, const Expression &aux) {
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));

  Expression x = in;

  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    assert(vars.size() >= L2H + 1);

    Expression y = vars[HB] + vars[X2H] * x + vars[L2H] * aux;

    if (t == 0 && h0.size() > 0) 
      y = y + vars[H2H] * h0[i];
    else if (t >= 1)
      y = y + vars[H2H] * h[t-1][i];

    x = h[t][i] = tanh(y);
  }
  return h[t].back();
}

} // namespace cnn
