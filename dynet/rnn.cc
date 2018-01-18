#include "dynet/rnn.h"

#include "dynet/expr.h"
#include "dynet/param-init.h"

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;
using namespace dynet;

namespace dynet {

enum { X2H=0, H2H, HB, L2H };

RNNBuilder::~RNNBuilder() {}

SimpleRNNBuilder::SimpleRNNBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model,
                       bool support_lags) : layers(layers), input_dim_(input_dim),hidden_dim_(hidden_dim),lagging(support_lags) {
  local_model = model.add_subcollection("simple-rnn-builder");
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    Parameter p_x2h = local_model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2h = local_model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_hb = local_model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    vector<Parameter> ps = {p_x2h, p_h2h, p_hb};
    if (lagging)
        ps.push_back(local_model.add_parameters({hidden_dim, hidden_dim}));
    params.push_back(ps);
    layer_input_dim = hidden_dim;
  }
  dropout_rate = dropout_rate_h = 0.f;
}

void SimpleRNNBuilder::new_graph_impl(ComputationGraph& cg, bool update) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    Parameter p_x2h = params[i][X2H];
    Parameter p_h2h = params[i][H2H];
    Parameter p_hb = params[i][HB];
    Expression i_x2h =  update ? parameter(cg,p_x2h) : const_parameter(cg,p_x2h);
    Expression i_h2h =  update ? parameter(cg,p_h2h) : const_parameter(cg,p_h2h);
    Expression i_hb =  update ? parameter(cg,p_hb) : const_parameter(cg,p_hb);
    vector<Expression> vars = {i_x2h, i_h2h, i_hb};

    if (lagging) {
        Parameter p_l2h = params[i][L2H];
        Expression i_l2h =  update ? parameter(cg,p_l2h) : const_parameter(cg,p_l2h);
        vars.push_back(i_l2h);
    }

    param_vars.push_back(vars);
  }
  _cg = &cg;
}

void SimpleRNNBuilder::start_new_sequence_impl(const vector<Expression>& h_0) {
  h.clear();
  h0 = h_0;
  DYNET_ARG_CHECK(h0.empty() || h0.size() == layers,
                          "Number of inputs passed to initialize RNNBuilder (" << h0.size() << ") is not equal to the number of layers (" << layers << ")");
  dropout_masks_valid = false;
}

Expression SimpleRNNBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
                          "Number of inputs passed to RNNBuilder::set_h() (" << h_new.size() << ") is not equal to the number of layers (" << layers << ")");
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression y = h_new[i];
    h[t][i] = y;
  }
  return h[t].back();
}

void SimpleRNNBuilder::set_dropout(float d){
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = dropout_rate_h = d;
}

void SimpleRNNBuilder::set_dropout(float d, float d_h){
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f && d_h >= 0.f && d_h <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
  dropout_rate_h = d_h;
}

void SimpleRNNBuilder::disable_dropout(){
  dropout_rate = dropout_rate_h = 0.f;
}

void SimpleRNNBuilder::set_dropout_masks(unsigned batch_size){
  masks.clear();
  for (unsigned i = 0; i < layers; ++i) {
    std::vector<Expression> masks_i;
    unsigned idim = (i == 0) ? input_dim_ : hidden_dim_;
    if (dropout_rate > 0.f || dropout_rate_h > 0.f) {
      float retention_rate = 1.f - dropout_rate;
      float retention_rate_h = 1.f - dropout_rate_h;
      float scale = 1.f / retention_rate;
      float scale_h = 1.f / retention_rate_h;

      // input
      masks_i.push_back(random_bernoulli(*_cg, Dim({idim}, batch_size), retention_rate, scale));

      // hidden
      masks_i.push_back(random_bernoulli(*_cg, Dim({hidden_dim_}, batch_size), retention_rate_h, scale_h));

      masks.push_back(masks_i);
    }
  }
  dropout_masks_valid = true;
}

Expression SimpleRNNBuilder::add_input_impl(int prev, const Expression &in) {
  if((dropout_rate != 0.f || dropout_rate_h !=0.f) && !dropout_masks_valid)
    set_dropout_masks(in.dim().bd);
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));

  Expression x = in;
  Expression h_prev;

  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];

    if(dropout_rate > 0.f){
      x = cmult(x,masks[i][0]);
    }

    bool exists_h_prev = false;
    // y <--- g(y_prev)
    if(prev >= 0) {
      h_prev = h[prev][i];
      exists_h_prev = true;
    } else if(h0.size() > 0) {
      h_prev = h0[i];
      exists_h_prev = true;
    }

    if(exists_h_prev) {
      if(dropout_rate_h > 0.f)
        h_prev = cmult(h_prev,masks[i][1]);
        x = h[t][i] = tanh( affine_transform({vars[2], vars[0], x, vars[1], h_prev}) );
    }
    else{
      x = h[t][i] = tanh( affine_transform({vars[2], vars[0], x}) );
    }
  } 
  return h[t].back();
}

Expression SimpleRNNBuilder::add_auxiliary_input(const Expression &in, const Expression &aux) {
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));

  Expression x = in;

  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    DYNET_ASSERT(vars.size() >= L2H + 1, "Failed dimension check in SimpleRNNBuilder");

    if(t > 0) {
      x = h[t][i] = tanh( affine_transform({vars[HB], vars[X2H], x, vars[L2H], aux, vars[H2H], h[t-1][i]}) );
    } else if(h0.size() > 0) {
      x = h[t][i] = tanh( affine_transform({vars[HB], vars[X2H], x, vars[L2H], aux, vars[H2H], h0[i]}) );
    } else {
      x = h[t][i] = tanh( affine_transform({vars[HB], vars[X2H], x, vars[L2H], aux}) );
    }

  }
  return h[t].back();
}

void SimpleRNNBuilder::copy(const RNNBuilder & rnn) {
  const SimpleRNNBuilder & rnn_simple = (const SimpleRNNBuilder&)rnn;
  DYNET_ARG_CHECK(params.size() == rnn_simple.params.size(),
                          "Attempt to copy between two SimpleRNNBuilders that are not the same size");
  for(size_t i = 0; i < rnn_simple.params.size(); ++i) {
    params[i][0] = rnn_simple.params[i][0];
    params[i][1] = rnn_simple.params[i][1];
    params[i][2] = rnn_simple.params[i][2];
  }
}

ParameterCollection & SimpleRNNBuilder::get_parameter_collection() {
  return local_model;
}

} // namespace dynet
