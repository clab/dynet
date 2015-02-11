#include "cnn/lstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/training.h"

using namespace std;

namespace cnn {

enum { X2I, H2I, C2I, BI, X2F, H2F, C2F, BF, X2O, H2O, C2O, BO, X2C, H2C, BC };

LSTMBuilder::~LSTMBuilder() {
  for (auto p : to_be_deleted) delete p;
}

LSTMBuilder::LSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Trainer* trainer) : layers(layers) {
  assert(layers < 10);
  // TODO move into stack-allocated object
  p_z = new ConstParameters(Matrix::Zero(hidden_dim, 1));
  to_be_deleted.push_back(p_z);

  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameters* p_x2i = new Parameters(Dim(hidden_dim, layer_input_dim));
    Parameters* p_h2i = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_c2i = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_bi = new Parameters(Dim(hidden_dim, 1));
    
    // f
    Parameters* p_x2f = new Parameters(Dim(hidden_dim, layer_input_dim));
    Parameters* p_h2f = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_c2f = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_bf = new Parameters(Dim(hidden_dim, 1));

    // o
    Parameters* p_x2o = new Parameters(Dim(hidden_dim, layer_input_dim));
    Parameters* p_h2o = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_c2o = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_bo = new Parameters(Dim(hidden_dim, 1));

    // c
    Parameters* p_x2c = new Parameters(Dim(hidden_dim, layer_input_dim));
    Parameters* p_h2c = new Parameters(Dim(hidden_dim, hidden_dim));
    Parameters* p_bc = new Parameters(Dim(hidden_dim, 1));
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    trainer->add_params({p_x2i, p_h2i, p_c2i, p_bi, p_x2f, p_h2f, p_c2f, p_bf, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc});
    vector<Parameters*> ps = {p_x2i, p_h2i, p_c2i, p_bi, p_x2f, p_h2f, p_c2f, p_bf, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc};
    params.push_back(ps);
    for (auto p : ps) to_be_deleted.push_back(p);
  }  // layers
}

void LSTMBuilder::add_parameter_edges(Hypergraph* hg) {
  zero = hg->add_input(p_z, "zero");
  param_vars.clear();
  h.clear();
  c.clear();
  for (unsigned i = 0; i < layers; ++i) {
    string layer = to_string(i);
    auto& p = params[i];

    // i
    unsigned i_x2i = hg->add_parameter(p[X2I], "x2i:" + layer);
    unsigned i_h2i = hg->add_parameter(p[H2I], "h2i:" + layer);
    unsigned i_c2i = hg->add_parameter(p[C2I], "c2i:" + layer);
    unsigned i_bi = hg->add_parameter(p[BI], "bi:" + layer);
    // f
    unsigned i_x2f = hg->add_parameter(p[X2F], "x2f:" + layer);
    unsigned i_h2f = hg->add_parameter(p[H2F], "h2f:" + layer);
    unsigned i_c2f = hg->add_parameter(p[C2F], "c2f:" + layer);
    unsigned i_bf = hg->add_parameter(p[BF], "bf:" + layer);
    // i
    unsigned i_x2o = hg->add_parameter(p[X2O], "x2o:" + layer);
    unsigned i_h2o = hg->add_parameter(p[H2O], "h2o:" + layer);
    unsigned i_c2o = hg->add_parameter(p[C2O], "c2o:" + layer);
    unsigned i_bo = hg->add_parameter(p[BO], "bo:" + layer);
    // c
    unsigned i_x2c = hg->add_parameter(p[X2C], "x2c:" + layer);
    unsigned i_h2c = hg->add_parameter(p[H2C], "h2c:" + layer);
    unsigned i_bc = hg->add_parameter(p[BC], "bc:" + layer);

    vector<unsigned> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2f, i_h2f, i_c2f, i_bf, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc};
    param_vars.push_back(vars);
  }
}

unsigned LSTMBuilder::add_input(unsigned x, Hypergraph* hg) {
  const unsigned t = h.size();
  string ts = to_string(t);
  h.push_back(vector<unsigned>(layers));
  c.push_back(vector<unsigned>(layers));
  vector<unsigned>& ht = h.back();
  vector<unsigned>& ct = c.back();
  unsigned in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<unsigned>& vars = param_vars[i];
    unsigned i_h_tm1 = t ? h[t-1][i] : zero;
    unsigned i_c_tm1 = t ? c[t-1][i] : zero;
    // input
    unsigned i_ait = hg->add_function<Multilinear>({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1}, "pi_" + ts);
    unsigned i_it = hg->add_function<LogisticSigmoid>({i_ait}, "i_" + ts);
    // forget
    unsigned i_aft = hg->add_function<Multilinear>({vars[BF], vars[X2F], in, vars[H2F], i_h_tm1, vars[C2F], i_c_tm1}, "pf_" + ts);
    unsigned i_ft = hg->add_function<LogisticSigmoid>({i_aft}, "f_" + ts);
    // write memory cell
    unsigned i_awt = hg->add_function<Multilinear>({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1}, "pc_" + ts);
    unsigned i_wt = hg->add_function<Tanh>({i_awt}, "w_" + ts);
    unsigned i_nwt = hg->add_function<CwiseMultiply>({i_it, i_wt}, "iw_" + ts);
    unsigned i_crt = hg->add_function<CwiseMultiply>({i_ft, i_c_tm1}, "cr_" + ts);
    ct[i] = hg->add_function<Sum>({i_crt, i_nwt}, "c_"+ts); // new memory cell at time t
 
    // output
    unsigned i_aot = hg->add_function<Multilinear>({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], ct[i]}, "po_" + ts);
    unsigned i_ot = hg->add_function<LogisticSigmoid>({i_aot}, "o_" + ts);
    unsigned ph_t = hg->add_function<Tanh>({ct[i]}, "ph_" + ts);
    ht[i] = hg->add_function<CwiseMultiply>({i_ot, ph_t}, "h_" + ts);
  }
  return ht.back();
}

} // namespace cnn
