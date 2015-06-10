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
    params.push_back(AffineBuilder(*model, {layer_input_dim, hidden_dim}, hidden_dim));
    layer_input_dim = hidden_dim;
  }
}

void SimpleRNNBuilder::new_graph_impl(ComputationGraph& cg) {
  for (unsigned i = 0; i < layers; ++i)
    params[i].add_to(cg);
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
    Expression y;
    if (t == 0 && h0.size() > 0)
      y = params[i]({x, h0[i]});
    else if (t > 0)
      y = params[i]({x, h[t-1][i]});
    else
      y = params[i]({x});

    x = h[t][i] = tanh(y);
  }
  return h[t].back();
}

} // namespace cnn
