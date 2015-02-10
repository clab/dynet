#include "cnn/rnn.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/training.h"

using namespace std;

namespace cnn {

RNNBuilder::RNNBuilder(Hypergraph* g,
                       unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Trainer* trainer) : hg(g), layers(layers) {
  assert(layers < 10);
  // TODO create free list
  ConstParameters* p_z = new ConstParameters(Matrix::Zero(hidden_dim, hidden_dim));
  zero = hg->add_input(p_z);

  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = new Parameters(Dim(hidden_dim, input_dim));
    Parameters* p_h2h = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_hb = new Parameters(Dim(hidden_dim, 1));
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
  h.push_back(vector<unsigned>(layers));
  vector<unsigned>& ht = h.back();
  unsigned in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<unsigned>& vars = param_vars[i];
    unsigned i_h_tm1 = t ? h[t-1][i] : zero;
    unsigned i_h1 = hg->add_function<MatrixMultiply>({vars[0], in}, "xxx");
    unsigned i_h2 = hg->add_function<MatrixMultiply>({vars[1], i_h_tm1}, "xxx");
    unsigned i_h3 = hg->add_function<Sum>({vars[2], i_h1, i_h2}, "xxx");
    in = ht[i] = hg->add_function<Tanh>({i_h3}, "xxx");
  }
  return ht.back();
}

} // namespace cnn
