#include "cnn/rnn.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/training.h"

using namespace std;

namespace cnn {

RNNBuilder::~RNNBuilder() {
  for (auto p : to_be_deleted) delete p;
}

RNNBuilder::RNNBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Trainer* trainer) : layers(layers) {
  assert(layers < 10);
  p_z = new ConstParameters(Matrix::Zero(hidden_dim, 1));
  to_be_deleted.push_back(p_z);

  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = new Parameters(Dim(hidden_dim, layer_input_dim));
    Parameters* p_h2h = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_hb = new Parameters(Dim(hidden_dim, 1));
    vector<Parameters*> ps = {p_x2h, p_h2h, p_hb};
    for (auto p : ps) to_be_deleted.push_back(p);
    params.push_back(ps);
    layer_input_dim = hidden_dim;
    trainer->add_params({p_x2h, p_h2h, p_hb});
  }
}

void RNNBuilder::add_parameter_edges(Hypergraph* hg) {
  zero = hg->add_input(p_z, "zero");
  param_vars.clear();
  h.clear();
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = params[i][0];
    Parameters* p_h2h = params[i][1];
    Parameters* p_hb = params[i][2];
    const string ts = to_string(i);
    unsigned i_x2h = hg->add_parameter(p_x2h, "x2h" + ts);
    unsigned i_h2h = hg->add_parameter(p_h2h, "h2h" + ts);
    unsigned i_hb = hg->add_parameter(p_hb, "hb" + ts);
    vector<unsigned> vars = {i_x2h, i_h2h, i_hb};
    param_vars.push_back(vars);
  }
}

unsigned RNNBuilder::add_input(unsigned x, Hypergraph* hg) {
  const unsigned t = h.size();
  string ts = to_string(t);
  h.push_back(vector<unsigned>(layers));
  vector<unsigned>& ht = h.back();
  unsigned in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<unsigned>& vars = param_vars[i];
    unsigned i_h_tm1 = t ? h[t-1][i] : zero;
    unsigned i_h3 = hg->add_function<Multilinear>({vars[2], vars[0], in, vars[1], i_h_tm1}, "ph_" + ts);
    in = ht[i] = hg->add_function<Tanh>({i_h3}, "h_" + ts);
  }
  return ht.back();
}

} // namespace cnn
