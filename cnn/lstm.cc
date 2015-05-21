#include "cnn/lstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"
#include "cnn/training.h"

using namespace std;

namespace cnn {

enum { X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC };

LSTMBuilder::LSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model* model) : layers(layers) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameters* p_x2i = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_c2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bi = model->add_parameters({hidden_dim});
    
    // o
    Parameters* p_x2o = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_c2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bo = model->add_parameters({hidden_dim});

    // c
    Parameters* p_x2c = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2c = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bc = model->add_parameters({hidden_dim});
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameters*> ps = {p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc};
    params.push_back(ps);
  }  // layers
}

void LSTMBuilder::new_graph_impl(ComputationGraph* cg) {
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i) {
    auto& p = params[i];

    // i
    VariableIndex i_x2i = cg->add_parameters(p[X2I]);
    VariableIndex i_h2i = cg->add_parameters(p[H2I]);
    VariableIndex i_c2i = cg->add_parameters(p[C2I]);
    VariableIndex i_bi = cg->add_parameters(p[BI]);

    // o
    VariableIndex i_x2o = cg->add_parameters(p[X2O]);
    VariableIndex i_h2o = cg->add_parameters(p[H2O]);
    VariableIndex i_c2o = cg->add_parameters(p[C2O]);
    VariableIndex i_bo = cg->add_parameters(p[BO]);

    // c
    VariableIndex i_x2c = cg->add_parameters(p[X2C]);
    VariableIndex i_h2c = cg->add_parameters(p[H2C]);
    VariableIndex i_bc = cg->add_parameters(p[BC]);

    vector<VariableIndex> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc};
    param_vars.push_back(vars);
  }
}

void LSTMBuilder::start_new_sequence_impl(const vector<VariableIndex>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      h0[i] = hinit[i];
      c0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }
}

VariableIndex LSTMBuilder::add_input_impl(VariableIndex x, ComputationGraph* cg) {
  const unsigned t = h.size();
  h.push_back(vector<VariableIndex>(layers));
  c.push_back(vector<VariableIndex>(layers));
  vector<VariableIndex>& ht = h.back();
  vector<VariableIndex>& ct = c.back();
  VariableIndex in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<VariableIndex>& vars = param_vars[i];
    VariableIndex i_h_tm1;
    VariableIndex i_c_tm1;
    bool has_prev_state = (t > 0 || has_initial_state);
    if (t == 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[t-1][i];
      i_c_tm1 = c[t-1][i];
    }
    // input
    VariableIndex i_ait;
    if (has_prev_state)
      i_ait = cg->add_function<AffineTransform>({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1});
    else
      i_ait = cg->add_function<AffineTransform>({vars[BI], vars[X2I], in});
    VariableIndex i_it = cg->add_function<LogisticSigmoid>({i_ait});
    // forget
    VariableIndex i_ft = cg->add_function<ConstantMinusX>({i_it}, 1.f);
    // write memory cell
    VariableIndex i_awt;
    if (has_prev_state)
      i_awt = cg->add_function<AffineTransform>({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
      i_awt = cg->add_function<AffineTransform>({vars[BC], vars[X2C], in});
    VariableIndex i_wt = cg->add_function<Tanh>({i_awt});
    // output
    if (has_prev_state) {
      VariableIndex i_nwt = cg->add_function<CwiseMultiply>({i_it, i_wt});
      VariableIndex i_crt = cg->add_function<CwiseMultiply>({i_ft, i_c_tm1});
      ct[i] = cg->add_function<Sum>({i_crt, i_nwt}); // new memory cell at time t
    } else {
      ct[i] = cg->add_function<CwiseMultiply>({i_it, i_wt});
    }
 
    VariableIndex i_aot;
    if (has_prev_state)
      i_aot = cg->add_function<AffineTransform>({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], ct[i]});
    else
      i_aot = cg->add_function<AffineTransform>({vars[BO], vars[X2O], in});
    VariableIndex i_ot = cg->add_function<LogisticSigmoid>({i_aot});
    VariableIndex ph_t = cg->add_function<Tanh>({ct[i]});
    in = ht[i] = cg->add_function<CwiseMultiply>({i_ot, ph_t});
  }
  return ht.back();
}

} // namespace cnn
