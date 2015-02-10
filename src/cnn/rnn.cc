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

RNNBuilder::RNNBuilder(Hypergraph* g,
                       unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Trainer* trainer) : hg(g), layers(layers) {
  assert(layers < 10);
  ConstParameters* p_z = new ConstParameters(Matrix::Zero(hidden_dim, hidden_dim));
  to_be_deleted.push_back(p_z);
  zero = hg->add_input(p_z, "zero");

  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = new Parameters(Dim(hidden_dim, layer_input_dim));
    Parameters* p_h2h = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_hb = new Parameters(Dim(hidden_dim, 1));
    layer_input_dim = hidden_dim;
    to_be_deleted.push_back(p_x2h);
    to_be_deleted.push_back(p_h2h);
    to_be_deleted.push_back(p_hb);
    trainer->add_params({p_x2h, p_h2h, p_hb});
    string x2h_name = "x2h0"; x2h_name[3] += i;
    unsigned i_x2h = hg->add_parameter(p_x2h, x2h_name);
    string h2h_name = "h2h0"; h2h_name[3] += i;
    unsigned i_h2h = hg->add_parameter(p_x2h, h2h_name);
    string hb_name = "b0"; h2h_name[2] += i;
    unsigned i_hb = hg->add_parameter(p_x2h, hb_name);
    vector<unsigned> vars = {i_x2h, i_h2h, i_hb};
    param_vars.push_back(vars);
  }
}

unsigned RNNBuilder::add_input(unsigned x) {
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
