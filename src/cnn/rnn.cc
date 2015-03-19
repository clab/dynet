#include "cnn/rnn.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/training.h"

using namespace std;

namespace cnn {

RNNBuilder::RNNBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model) : hidden_dim(hidden_dim), layers(layers) {
  builder_state = 0; // created
  assert(layers < 10);

  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = model->add_parameters(Dim(hidden_dim, layer_input_dim));
    Parameters* p_h2h = model->add_parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_hb = model->add_parameters(Dim(hidden_dim, 1));
    vector<Parameters*> ps = {p_x2h, p_h2h, p_hb};
    params.push_back(ps);
    layer_input_dim = hidden_dim;
  }
}

void RNNBuilder::new_graph() {
  param_vars.clear();
  h.clear();
  h0.clear();
  builder_state = 1;
}

void RNNBuilder::add_parameter_edges(Hypergraph* hg) {
  if (builder_state != 1) {
    cerr << "Invalid state: " << builder_state << endl;
    abort();
  }
  builder_state = 2;

  if (h0.empty()) {
    VariableIndex zero_input = hg->add_input(Matrix::Zero(hidden_dim, 1));
    h0 = vector<VariableIndex>(layers, zero_input);
  }

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

void RNNBuilder::add_parameter_edges(Hypergraph* hg, vector<VariableIndex> h_0) {
  h0 = h_0;
  add_parameter_edges(hg);
  assert (h0.size() == layers);
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
    VariableIndex i_h_tm1;
    if (t == 0) {  // first time step
      // initial value of h for layer i at timestep 0
      // defaults to zero matrix if not set in add_parameter_edges
      i_h_tm1 = h0[i];
    } else {  // tth time step
      i_h_tm1 = h[t-1][i];
    }
    // h3 = hbias + h2h * h_{t-1} + x2h * in
    VariableIndex i_h3 = hg->add_function<Multilinear>({vars[2], vars[0], in, vars[1], i_h_tm1});
    in = ht[i] = hg->add_function<Tanh>({i_h3});
  }
  return ht.back();
}

} // namespace cnn
