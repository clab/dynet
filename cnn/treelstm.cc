#include "cnn/treelstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

enum { X2I, C2I, BI, X2F, C2F, BF, X2O, C2O, BO, X2C, BC };
enum { H2I, H2F, H2O, H2C };
// See "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
// by Tai, Socher, and Manning (2015), section 3.2, for details on this model.
// http://arxiv.org/pdf/1503.00075v3.pdf
TreeLSTMBuilder::TreeLSTMBuilder(unsigned N,
                         unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model* model) : layers(layers), N(N) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameters* p_x2i = model->add_parameters({hidden_dim, layer_input_dim});
    LookupParameters* p_h2i = model->add_lookup_parameters(N, {hidden_dim, hidden_dim});
    Parameters* p_c2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bi = model->add_parameters({hidden_dim});

    // f
    Parameters* p_x2f = model->add_parameters({hidden_dim, layer_input_dim});
    LookupParameters* p_h2f = model->add_lookup_parameters(N*N, {hidden_dim, hidden_dim});
    Parameters* p_c2f = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bf = model->add_parameters({hidden_dim});
    
    // o
    Parameters* p_x2o = model->add_parameters({hidden_dim, layer_input_dim});
    LookupParameters* p_h2o = model->add_lookup_parameters(N, {hidden_dim, hidden_dim});
    Parameters* p_c2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bo = model->add_parameters({hidden_dim});

    // c (a.k.a. u)
    Parameters* p_x2c = model->add_parameters({hidden_dim, layer_input_dim});
    LookupParameters* p_h2c = model->add_lookup_parameters(N, {hidden_dim, hidden_dim});
    Parameters* p_bc = model->add_parameters({hidden_dim});
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameters*> ps = {p_x2i, p_c2i, p_bi, p_x2f, p_c2f, p_bf, p_x2o, p_c2o, p_bo, p_x2c, p_bc};
    vector<LookupParameters*> lps = {p_h2i, p_h2f, p_h2o, p_h2c};
    params.push_back(ps);
    lparams.push_back(lps);
  }  // layers
}

void TreeLSTMBuilder::new_graph_impl(ComputationGraph& cg){
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];
    auto& lp = lparams[i];

    //i
    Expression i_x2i = parameter(cg, p[X2I]);
    Expression i_c2i = parameter(cg, p[C2I]);
    Expression i_bi = parameter(cg, p[BI]);
    //f
    Expression i_x2f = parameter(cg, p[X2F]);
    Expression i_c2f = parameter(cg, p[C2F]);
    Expression i_bf = parameter(cg, p[BF]);
    //o
    Expression i_x2o = parameter(cg, p[X2O]);
    Expression i_c2o = parameter(cg, p[C2O]);
    Expression i_bo = parameter(cg, p[BO]);
    //c
    Expression i_x2c = parameter(cg, p[X2C]);
    Expression i_bc = parameter(cg, p[BC]);

    vector<Expression> vars = {i_x2i, i_c2i, i_bi, i_x2f, i_c2f, i_bf, i_x2o, i_c2o, i_bo, i_x2c, i_bc};
    param_vars.push_back(vars);

    vector<Expression> lvars((3 + N) * N);
    for (unsigned k = 0; k < N; ++k) {
      lvars[k * (N + 3) + 0] = lookup(cg, lp[H2I], k);
      lvars[k * (N + 3) + 1] = lookup(cg, lp[H2O], k);
      lvars[k * (N + 3) + 2] = lookup(cg, lp[H2C], k);
      for (unsigned j = 0; j < N; ++j) {
        lvars[k * (N + 3) + 3 + j] = lookup(cg, lp[H2F], k);
      }
    }
    lparam_vars.push_back(lvars); 
  }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void TreeLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
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

Expression TreeLSTMBuilder::add_input(vector<int> children, const Expression& x) {
  ComputationGraph& cg = *(ComputationGraph*)x.pg;
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();

  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    vector<Expression> i_h_children, i_c_children;
    i_h_children.reserve(children.size() > 1 ? children.size() : 1);
    i_c_children.reserve(children.size() > 1 ? children.size() : 1); 

    bool has_prev_state = (children.size() > 0 || has_initial_state);
    if (children.size() == 0) {
      i_h_children.push_back(Expression());
      i_c_children.push_back(Expression());
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_children[0] = h0[i];
        i_c_children[0] = c0[i];
      }
    }
    else {  // t > 0
      for (int child : children) {
        i_h_children.push_back(h[child][i]);
        i_c_children.push_back(c[child][i]);
      }
    }

    // input
    Expression i_ait;
    if (has_prev_state) {
      vector<Expression> xs = {vars[BI], vars[X2I], in};
      xs.reserve(2 * children.size() + 3);
      for (unsigned j = 0; j < children.size(); ++j) {
        unsigned ej = (j < N) ? j : N - 1;
        xs.push_back(lookup(cg, lparams[i][H2I], ej));
        //xs.push_back(lparam_vars[i][j * (N + 3) + 0]);
        xs.push_back(i_h_children[j]);
      }
      assert (xs.size() == 2 * children.size() + 3);
      i_ait = affine_transform(xs);
    }
    else
      i_ait = affine_transform({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);

    // forget 
    vector<Expression> i_ft;
    for (unsigned k = 0; k < children.size(); ++k) {
      unsigned ek = (k < N) ? k : N - 1;
      Expression i_aft;
      if (has_prev_state) {
        vector<Expression> xs = {vars[BF], vars[X2F], in};
        xs.reserve(2 * children.size() + 3);
        for (unsigned j = 0; j < children.size(); ++j) {
          unsigned ej = (j < N) ? j : N - 1;
          xs.push_back(lookup(cg, lparams[i][H2F], ej * N + ek));
          //xs.push_back(lparam_vars[i][j * (N + 3) + 3 + k]);
          xs.push_back(i_h_children[j]);
        }
        assert (xs.size() == 2 * children.size() + 3);
        i_aft = affine_transform(xs);
      }
      else
        i_ait = affine_transform({vars[BF], vars[X2F], in});
      i_ft.push_back(logistic(i_aft));
    }

    // write memory cell
    Expression i_awt;
    if (has_prev_state) {
      vector<Expression> xs = {vars[BC], vars[X2C], in};
      // This is the one and only place that should *not* condition on i_c_children
      // This should condition only on x (a.k.a. in), the bias (vars[BC]) and i_h_children
      xs.reserve(2 * children.size() + 3);
      for (unsigned j = 0; j < children.size(); ++j) {
        unsigned ej = (j < N) ? j : N - 1;
        xs.push_back(lookup(cg, lparams[i][H2C], ej));
        //xs.push_back(lparam_vars[i][j * (N + 3) + 2]);
        xs.push_back(i_h_children[j]);
      }
      assert (xs.size() == 2 * children.size() + 3);
      i_awt = affine_transform(xs);
    }
    else
      i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);

    // compute new cell value
    if (has_prev_state) {
      Expression i_nwt = cwise_multiply(i_it, i_wt);
      vector<Expression> i_crts(children.size());
      for (unsigned j = 0; j < children.size(); ++j) {
        i_crts[j] = cwise_multiply(i_ft[j], i_c_children[j]);
      }
      Expression i_crt = sum(i_crts);
      ct[i] = i_crt + i_nwt;
    }
    else {
      ct[i] = cwise_multiply(i_it, i_wt);
    }
 
    // output
    Expression i_aot;
    if (has_prev_state) {
      vector<Expression> xs = {vars[BO], vars[X2O], in};
      xs.reserve(2 * children.size() + 3);
      for (unsigned j = 0; j < children.size(); ++j) {
        unsigned ej = (j < N) ? j : N - 1;
        xs.push_back(lookup(cg, lparams[i][H2O], ej));
        //xs.push_back(lparam_vars[i][j * (N + 3) + 1]);
        xs.push_back(i_h_children[j]);
      }
      assert (xs.size() == 2 * children.size() + 3);
      i_aot = affine_transform(xs);
    }
    else
      i_aot = affine_transform({vars[BO], vars[X2O], in});
    Expression i_ot = logistic(i_aot);

    // Compute new h value
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cwise_multiply(i_ot, ph_t);
  }
  return ht.back();
}

Expression TreeLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  assert (false);
  return x;
}

void TreeLSTMBuilder::copy(const RNNBuilder & rnn) {
  const TreeLSTMBuilder & rnn_treelstm = (const TreeLSTMBuilder&)rnn;
  assert(params.size() == rnn_treelstm.params.size());
  for(size_t i = 0; i < params.size(); ++i)
      for(size_t j = 0; j < params[i].size(); ++j)
        params[i][j]->copy(*rnn_treelstm.params[i][j]);
}

} // namespace cnn
