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
  builder_state = 0; // created
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

void RNNBuilder::new_graph() {
  param_vars.clear();
  h.clear();
  builder_state = 1;  
}

void RNNBuilder::add_parameter_edges(Hypergraph* hg) {
  zero = hg->add_input(p_z);
  if (builder_state != 1) {
    cerr << "Invalid state: " << builder_state << endl;
    abort();
  }
  builder_state = 2;
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = params[i][0];
    Parameters* p_h2h = params[i][1];
    Parameters* p_hb = params[i][2];
    const string ts = to_string(i);
    VariableIndex i_x2h = hg->add_parameter(p_x2h);
    VariableIndex i_h2h = hg->add_parameter(p_h2h);
    VariableIndex i_hb = hg->add_parameter(p_hb);
    vector<VariableIndex> vars = {i_x2h, i_h2h, i_hb};
    param_vars.push_back(vars);
  }
}

VariableIndex RNNBuilder::add_input(VariableIndex x, Hypergraph* hg) {
  if (builder_state != 2) {
    cerr << "Invalid state: " << builder_state << endl;
    abort();
  }
  const unsigned t = h.size();
  string ts = to_string(t);
  h.push_back(vector<VariableIndex>(layers));
  vector<VariableIndex>& ht = h.back();
  VariableIndex in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<VariableIndex>& vars = param_vars[i];
    VariableIndex i_h_tm1 = t ? h[t-1][i] : zero;
    VariableIndex i_h3 = hg->add_function<Multilinear>({vars[2], vars[0], in, vars[1], i_h_tm1});
    in = ht[i] = hg->add_function<Tanh>({i_h3});
  }
  return ht.back();
}

} // namespace cnn
