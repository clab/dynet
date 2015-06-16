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

RNNBuilder::~RNNBuilder() {}

SimpleRNNBuilder::SimpleRNNBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model) : layers(layers) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2h = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_hb = model->add_parameters({hidden_dim});
    vector<Parameters*> ps = {p_x2h, p_h2h, p_hb};
    params.push_back(ps);
    layer_input_dim = hidden_dim;
  }
}

void SimpleRNNBuilder::new_graph_impl(ComputationGraph& cg) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = params[i][0];
    Parameters* p_h2h = params[i][1];
    Parameters* p_hb = params[i][2];
    Expression i_x2h =  parameter(cg,p_x2h);
    Expression i_h2h =  parameter(cg,p_h2h);
    Expression i_hb =  parameter(cg,p_hb);

    vector<Expression> vars = {i_x2h, i_h2h, i_hb};
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

    Expression y = vars[2] + vars[0] * x;

    if (t == 0 && h0.size() > 0)
      y = y + vars[1] * h0[i];
    else if (t > 0)
      y = y + vars[1] * h[t-1][i];

    x = h[t][i] = tanh(y);
  }
  return h[t].back();
}

} // namespace cnn
