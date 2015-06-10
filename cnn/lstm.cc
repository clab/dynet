#include "cnn/lstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

LSTMBuilder::LSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model* model) : layers(layers) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    i_params.push_back(AffineBuilder(*model, {layer_input_dim, hidden_dim, hidden_dim}, hidden_dim));
    o_params.push_back(AffineBuilder(*model, {layer_input_dim, hidden_dim, hidden_dim}, hidden_dim));
    c_params.push_back(AffineBuilder(*model, {layer_input_dim, hidden_dim}, hidden_dim));
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next
  }  // layers
}

void LSTMBuilder::new_graph_impl(ComputationGraph& cg) {
  for (unsigned i = 0; i < layers; ++i) {
    i_params[i].add_to(cg);
    o_params[i].add_to(cg);
    c_params[i].add_to(cg);
  }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void LSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }
}

Expression LSTMBuilder::add_input_impl(const Expression& x) {
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {
    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (t > 0 || has_initial_state);
    if (t == 0) {
      if (has_initial_state) {
        // initial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[t-1][i];
      i_c_tm1 = c[t-1][i];
    }
    // input
    Expression i_ait;
    if (has_prev_state)
      i_ait = i_params[i]({in, i_h_tm1, i_c_tm1});
    else
      i_ait = i_params[i]({in});
    Expression i_it = logistic(i_ait);

    // write memory cell
    Expression i_awt;
    if (has_prev_state)
      i_awt = c_params[i]({in, i_h_tm1});
    else
      i_awt = c_params[i]({in});
    Expression i_wt = tanh(i_awt);

    // output
    if (has_prev_state) {
      ct[i] = cwise_multiply(i_it, i_wt) + cwise_multiply(1.f-i_it, i_c_tm1);
    } else {
      ct[i] = cwise_multiply(i_it, i_wt);
    }
 
    Expression i_aot;
    if (has_prev_state)
      i_aot = o_params[i]({in, i_h_tm1, ct[i]});
    else
      i_aot = o_params[i]({in});

    in = ht[i] = cwise_multiply(logistic(i_aot), tanh(ct[i]));
  }
  return ht.back();
}

} // namespace cnn
