#include "dynet/lstm.h"

#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "dynet/nodes.h"
#include "dynet/io-macros.h"

using namespace std;
using namespace dynet::expr;

namespace dynet {

enum { X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC };
enum { IA, IO, IW, HA, HO, HW, CA, CO, CW }; // Gal dropout masks

LSTMBuilder::LSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model* model) : layers(layers), input_dim(input_dim), hidden_dim(hidden_dim) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameter p_x2i = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bi = model->add_parameters({hidden_dim});

    // o
    Parameter p_x2o = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bo = model->add_parameters({hidden_dim});

    // c
    Parameter p_x2c = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2c = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bc = model->add_parameters({hidden_dim});
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameter> ps = {p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc};
    params.push_back(ps);
  }  // layers
  dropout_rate = 0.f;
}

void LSTMBuilder::new_graph_impl(ComputationGraph& cg) {
    _cg = &cg;
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i) {
    auto& p = params[i];

    //i
    Expression i_x2i = parameter(cg, p[X2I]);
    Expression i_h2i = parameter(cg, p[H2I]);
    Expression i_c2i = parameter(cg, p[C2I]);
    Expression i_bi = parameter(cg, p[BI]);
    //o
    Expression i_x2o = parameter(cg, p[X2O]);
    Expression i_h2o = parameter(cg, p[H2O]);
    Expression i_c2o = parameter(cg, p[C2O]);
    Expression i_bo = parameter(cg, p[BO]);
    //c
    Expression i_x2c = parameter(cg, p[X2C]);
    Expression i_h2c = parameter(cg, p[H2C]);
    Expression i_bc = parameter(cg, p[BC]);

    vector<Expression> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc};
    param_vars.push_back(vars);
  }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void LSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  // Init Gal Dropout masks for each layer
  masks.clear();
  for (unsigned i = 0; i < layers; ++i) {
    std::vector<Expression> masks_i;
    unsigned idim = (i == 0) ? input_dim : hidden_dim;
    if (dropout_rate > 0.f) {
      float retention_rate = 1.f - dropout_rate;
      float scale = 1.f / retention_rate;
      // in
      masks_i.push_back(random_bernoulli(*_cg,{ idim}, retention_rate) * scale);
      masks_i.push_back(random_bernoulli(*_cg,{ idim}, retention_rate * scale));
      masks_i.push_back(random_bernoulli(*_cg,{ idim}, retention_rate * scale));
      // h
      masks_i.push_back(random_bernoulli(*_cg,{ hidden_dim}, retention_rate * scale));
      masks_i.push_back(random_bernoulli(*_cg,{ hidden_dim}, retention_rate * scale));
      masks_i.push_back(random_bernoulli(*_cg,{ hidden_dim}, retention_rate * scale));
      // c
      masks_i.push_back(random_bernoulli(*_cg,{ hidden_dim}, retention_rate * scale));
      masks_i.push_back(random_bernoulli(*_cg,{ hidden_dim}, retention_rate * scale));
      masks_i.push_back(random_bernoulli(*_cg,{ hidden_dim}, retention_rate * scale));
      masks.push_back(masks_i);
    }
  }
  if (hinit.size() > 0) {
    assert(layers * 2 == hinit.size());
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

// TO DO - Make this correct
// Copied c from the previous step (otherwise c.size()< h.size())
// Also is creating a new step something we want? 
// wouldn't overwriting the current one be better?
Expression LSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  if (h_new.size()) { assert(h_new.size() == layers); }
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = h_new[i];
    Expression c_i = c[t - 1][i];
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}
// Current implementation : s_new is either {new_c[0],...,new_c[n]}
// or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
Expression LSTMBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
  if (s_new.size()) { assert(s_new.size() == layers || s_new.size() == 2 * layers ); }
  bool only_c = s_new.size() == layers;
  const unsigned t = c.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = only_c ? h[t - 1][i] : s_new[i + layers];
    Expression c_i = s_new[i];
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}

Expression LSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    const vector<Expression>& masks_i = masks[i];
    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }
    // apply dropout according to https://arxiv.org/abs/1512.05287
    // input
    Expression i_ait;
    if (has_prev_state)
      if (dropout_rate > 0.f)
        i_ait = affine_transform({vars[BI], vars[X2I], cmult(in, masks_i[IA]) , vars[H2I], cmult(i_h_tm1, masks_i[HA]), vars[C2I], cmult(i_c_tm1, masks_i[CA])});
      else
        i_ait = affine_transform({vars[BI], vars[X2I], in , vars[H2I], i_h_tm1, vars[C2I], i_c_tm1});
    else
      if (dropout_rate > 0.f)
        i_ait = affine_transform({vars[BI], vars[X2I], cmult(in, masks_i[IA])});
      else
        i_ait = affine_transform({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);
    // forget
    Expression i_ft = 1.f - i_it;
    // write memory cell
    Expression i_awt;
    if (has_prev_state)
      if (dropout_rate > 0.f)
        i_awt = affine_transform({vars[BC], vars[X2C], cmult(in, masks_i[IW]), vars[H2C], cmult(i_h_tm1, masks_i[HW])});
      else
        i_awt = affine_transform({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
      if (dropout_rate > 0.f)
        i_awt = affine_transform({vars[BC], vars[X2C], cmult(in, masks_i[IW])});
      else
        i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);
    // output
    if (has_prev_state) {
      Expression i_nwt = cmult(i_it, i_wt);
      Expression i_crt = cmult(i_ft, i_c_tm1);
      ct[i] = i_crt + i_nwt;
    } else {
      ct[i] = cmult(i_it, i_wt);
    }

    Expression i_aot;
    if (has_prev_state)
      if (dropout_rate > 0.f)
        i_aot = affine_transform({vars[BO], vars[X2O], cmult(in, masks_i[IO]), vars[H2O], cmult(i_h_tm1, masks_i[HO]), vars[C2O], cmult(ct[i], masks_i[CO])});
      else
        i_aot = affine_transform({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], ct[i]});
    else
      if (dropout_rate > 0.f)
        i_aot = affine_transform({vars[BO], vars[X2O], cmult(in, masks_i[IO]), vars[C2O], cmult(ct[i], masks_i[CO])});
      else
        i_aot = affine_transform({vars[BO], vars[X2O], in, vars[C2O], ct[i]});
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cmult(i_ot, ph_t);
  }
  return ht.back();
}

void LSTMBuilder::copy(const RNNBuilder & rnn) {
  const LSTMBuilder & rnn_lstm = (const LSTMBuilder&)rnn;
  assert(params.size() == rnn_lstm.params.size());
  for (size_t i = 0; i < params.size(); ++i)
    for (size_t j = 0; j < params[i].size(); ++j)
      params[i][j] = rnn_lstm.params[i][j];
}

void LSTMBuilder::save_parameters_pretraining(const string& fname) const {
  cerr << "Writing LSTM parameters to " << fname << endl;
  ofstream of(fname);
  assert(of);
  boost::archive::binary_oarchive oa(of);
  std::string id = "LSTMBuilder:params";
  oa << id;
  oa << layers;
  for (unsigned i = 0; i < layers; ++i) {
    for (auto p : params[i]) {
      oa << p.get()->values;
    }
  }
}

void LSTMBuilder::load_parameters_pretraining(const string& fname) {
  cerr << "Loading LSTM parameters from " << fname << endl;
  ifstream of(fname);
  assert(of);
  boost::archive::binary_iarchive ia(of);
  std::string id;
  ia >> id;
  if (id != "LSTMBuilder:params") {
    cerr << "Bad id read\n";
    abort();
  }
  unsigned l = 0;
  ia >> l;
  if (l != layers) {
    cerr << "Bad number of layers\n";
    abort();
  }
  // TODO check other dimensions
  for (unsigned i = 0; i < layers; ++i) {
    for (auto p : params[i]) {
      ia >> p.get()->values;
    }
  }
}

template<class Archive>
void LSTMBuilder::serialize(Archive& ar, const unsigned int) {
  ar & boost::serialization::base_object<RNNBuilder>(*this);
  ar & params;
  ar & layers;
  ar & dropout_rate;
}
DYNET_SERIALIZE_IMPL(LSTMBuilder);

} // namespace dynet
