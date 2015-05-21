#include "cnn/gru.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"
#include "cnn/training.h"

using namespace std;

namespace cnn {

enum { X2Z, H2Z, BZ, X2R, H2R, BR, X2H, H2H, BH };

GRUBuilder::GRUBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model) : hidden_dim(hidden_dim), layers(layers), zeros(hidden_dim, 0) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // z
    Parameters* p_x2z = model->add_parameters(Dim({hidden_dim, layer_input_dim}));
    Parameters* p_h2z = model->add_parameters(Dim({hidden_dim, hidden_dim}));
    Parameters* p_bz = model->add_parameters(Dim({hidden_dim}));
    
    // r
    Parameters* p_x2r = model->add_parameters(Dim({hidden_dim, layer_input_dim}));
    Parameters* p_h2r = model->add_parameters(Dim({hidden_dim, hidden_dim}));
    Parameters* p_br = model->add_parameters(Dim({hidden_dim}));

    // h
    Parameters* p_x2h = model->add_parameters(Dim({hidden_dim, layer_input_dim}));
    Parameters* p_h2h = model->add_parameters(Dim({hidden_dim, hidden_dim}));
    Parameters* p_bh = model->add_parameters(Dim({hidden_dim}));
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameters*> ps = {p_x2z, p_h2z, p_bz, p_x2r, p_h2r, p_br, p_x2h, p_h2h, p_bh};
    params.push_back(ps);
  }  // layers
}

void GRUBuilder::new_graph(ComputationGraph* hg) {
  sm.transition(RNNOp::new_graph);
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i) {
    string layer = to_string(i);
    auto& p = params[i];

    // z
    VariableIndex i_x2z = hg->add_parameters(p[X2Z]);
    VariableIndex i_h2z = hg->add_parameters(p[H2Z]);
    VariableIndex i_bz = hg->add_parameters(p[BZ]);

    // r
    VariableIndex i_x2r = hg->add_parameters(p[X2R]);
    VariableIndex i_h2r = hg->add_parameters(p[H2R]);
    VariableIndex i_br = hg->add_parameters(p[BR]);

    // h
    VariableIndex i_x2h = hg->add_parameters(p[X2H]);
    VariableIndex i_h2h = hg->add_parameters(p[H2H]);
    VariableIndex i_bh = hg->add_parameters(p[BH]);

    vector<VariableIndex> vars = {i_x2z, i_h2z, i_bz, i_x2r, i_h2r, i_br, i_x2h, i_h2h, i_bh};
    param_vars.push_back(vars);
  }
}

void GRUBuilder::start_new_sequence(ComputationGraph* hg,
                                    vector<VariableIndex> h_0) {
  sm.transition(RNNOp::start_new_sequence);
  h.clear();
  h0 = h_0;
  if (h0.empty()) {
    VariableIndex zero_input = hg->add_input(Dim({hidden_dim}), &zeros);
    if (h0.empty()) { h0 = vector<VariableIndex>(layers, zero_input); }
  }
  assert (h0.size() == layers);
}

VariableIndex GRUBuilder::add_input(VariableIndex x, ComputationGraph* hg) {
  sm.transition(RNNOp::add_input);
  const unsigned t = h.size();
  h.push_back(vector<VariableIndex>(layers));
  vector<VariableIndex>& ht = h.back();
  VariableIndex in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<VariableIndex>& vars = param_vars[i];
    VariableIndex i_h_tm1;
    if (t == 0) {
      // intial value for h at timestep 0 in layer i
      // defaults to zero matrix input if not set in add_parameter_edges
      i_h_tm1 = h0[i];
    } else {  // t > 0
      i_h_tm1 = h[t-1][i];
    }
    // update gate
    VariableIndex i_zt = hg->add_function<AffineTransform>({vars[BZ], vars[X2Z], in, vars[H2Z], i_h_tm1});
    i_zt = hg->add_function<LogisticSigmoid>({i_zt});
    // forget
    VariableIndex i_ft = hg->add_function<ConstantMinusX>({i_zt}, 1.f);
    // reset gate
    VariableIndex i_rt = hg->add_function<AffineTransform>({vars[BR], vars[X2R], in, vars[H2R], i_h_tm1});
    i_rt = hg->add_function<LogisticSigmoid>({i_rt});
    // candidate activation
    VariableIndex i_ght = hg->add_function<CwiseMultiply>({i_rt, i_h_tm1});
    VariableIndex i_ct = hg->add_function<AffineTransform>({vars[BH], vars[X2H], in, vars[H2H], i_ght});
    i_ct = hg->add_function<Tanh>({i_ct});

    // new hidden state
    VariableIndex i_nwt = hg->add_function<CwiseMultiply>({i_zt, i_ct});
    VariableIndex i_crt = hg->add_function<CwiseMultiply>({i_ft, i_h_tm1});
    in = ht[i] = hg->add_function<Sum>({i_crt, i_nwt});
  }
  return ht.back();
}

} // namespace cnn
