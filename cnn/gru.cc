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
                       Model* model) : hidden_dim(hidden_dim), layers(layers) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // z
    Parameters* p_x2z = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2z = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bz = model->add_parameters({hidden_dim});

    // r
    Parameters* p_x2r = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2r = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_br = model->add_parameters({hidden_dim});

    // h
    Parameters* p_x2h = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2h = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bh = model->add_parameters({hidden_dim});
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameters*> ps = {p_x2z, p_h2z, p_bz, p_x2r, p_h2r, p_br, p_x2h, p_h2h, p_bh};
    params.push_back(ps);
  }  // layers
}

void GRUBuilder::new_graph_impl(ComputationGraph& cg) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    auto& p = params[i];

    // z
    Expression x2z = parameter(cg,p[X2Z]);
    Expression h2z = parameter(cg,p[H2Z]);
    Expression bz = parameter(cg,p[BZ]);

    // r
    Expression x2r = parameter(cg,p[X2R]);
    Expression h2r = parameter(cg,p[H2R]);
    Expression br = parameter(cg,p[BR]);

    // h
    Expression x2h = parameter(cg,p[X2H]);
    Expression h2h = parameter(cg,p[H2H]);
    Expression bh = parameter(cg,p[BH]);

    vector<Expression> vars = {x2z, h2z, bz, x2r, h2r, br, x2h, h2h, bh};
    param_vars.push_back(vars);
  }
}

void GRUBuilder::start_new_sequence_impl(const std::vector<Expression>& h_0) {
  h.clear();
  h0 = h_0;
  if (!h0.empty()) {
    assert (h0.size() == layers);
  }
}

Expression GRUBuilder::add_input_impl(int prev, const Expression& x) {
  const bool has_initial_state = (h0.size() > 0);
  h.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression h_tprev;
    // prev_zero means that h_tprev should be treated as 0
    bool prev_zero = false;
    if (prev >= 0 || has_initial_state) {
      h_tprev = (prev < 0) ? h0[i] : h[prev][i];
    } else { prev_zero = true; }
    // update gate
    Expression zt;
    if (prev_zero)
      zt = affine_transform({vars[BZ], vars[X2Z], in});
    else
      zt = affine_transform({vars[BZ], vars[X2Z], in, vars[H2Z], h_tprev});
    zt = logistic(zt);
    // forget
    Expression ft = 1.f - zt;
    // reset gate
    Expression rt;
    if (prev_zero)
      rt = affine_transform({vars[BR], vars[X2R], in});
    else
      rt = affine_transform({vars[BR], vars[X2R], in, vars[H2R], h_tprev});
    rt = logistic(rt);

    // candidate activation
    Expression ct;
    if (prev_zero) {
      ct = affine_transform({vars[BH], vars[X2H], in});
      ct = tanh(ct);
      Expression nwt = cwise_multiply(zt, ct);
      in = ht[i] = nwt;
    } else {
      Expression ght = cwise_multiply(rt, h_tprev);
      ct = affine_transform({vars[BH], vars[X2H], in, vars[H2H], ght});
      ct = tanh(ct);
      Expression nwt = cwise_multiply(zt, ct);
      Expression crt = cwise_multiply(ft, h_tprev);
      in = ht[i] = crt + nwt;
    }
  }
  return ht.back();
}

void GRUBuilder::copy(const RNNBuilder & rnn) {
  const GRUBuilder & rnn_gru = (const GRUBuilder&)rnn;
  assert(params.size() == rnn_gru.params.size());
  for(size_t i = 0; i < params.size(); ++i)
      for(size_t j = 0; j < params[i].size(); ++j)
        params[i][j]->copy(*rnn_gru.params[i][j]);
}

} // namespace cnn
