#include "cnn/lstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

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

void LSTMBuilder::new_graph_impl(ComputationGraph& cg){
  param_vars.clear();
  cerr << "LSTMBuilder: About to build graph"<< endl;  

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];

    //i
    Expression i_x2i = parameter(cg,p[X2I]);
    Expression i_h2i = parameter(cg,p[H2I]);
    Expression i_c2i = parameter(cg,p[C2I]);
    Expression i_bi = parameter(cg,p[BI]);
    //o
    Expression i_x2o = parameter(cg,p[X2O]);
    Expression i_h2o = parameter(cg,p[H2O]);
    Expression i_c2o = parameter(cg,p[C2O]);
    Expression i_bo = parameter(cg,p[BO]);
    //c
    Expression i_x2c = parameter(cg,p[X2C]);
    Expression i_h2c = parameter(cg,p[H2C]);
    Expression i_bc = parameter(cg,p[BC]);

    vector<Expression> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc};
    param_vars.push_back(vars);
    
    
  }
  cerr << "LSTMBuilder: Built graph"<< endl;  
}

/*void LSTMBuilder::new_graph_impl(ComputationGraph* cg) {
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
}*/

// layout: 0..layers = c
//         layers+1..2*layers = h
void LSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
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

//VariableIndex LSTMBuilder::add_input_impl(VariableIndex x, ComputationGraph* cg) {
Expression LSTMBuilder::add_input_impl(Expression& x, ComputationGraph& cg) {
  cerr << "Input_impl: Came here!" << endl;
  const unsigned t = h.size();
  cerr << "Input_impl: Got size" << endl;
  //h.push_back(vector<VariableIndex>(layers));
  //c.push_back(vector<VariableIndex>(layers));
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  cerr << "Input_impl: pushed back" << endl;
  //vector<VariableIndex>& ht = h.back();
  //vector<VariableIndex>& ct = c.back();
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  //VariableIndex in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    //VariableIndex i_h_tm1;
    //VariableIndex i_c_tm1;
    Expression i_h_tm1, i_c_tm1;
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
    Expression i_ait;
    if (has_prev_state)
      // COmbine VariableIndex with Expressions???
      i_ait = vars[BI] + vars[X2I] * in + vars[H2I]*i_h_tm1 + vars[C2I] * i_c_tm1;
      //i_ait = cg->add_function<AffineTransform>({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1});
    else
      i_ait = vars[BI] + vars[X2I] * in;
      //i_ait = cg->add_function<AffineTransform>({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);
    //VariableIndex i_it = cg->add_function<LogisticSigmoid>({i_ait});
    // forget
    Expression i_ft = i_it - 1.f;
    //VariableIndex i_ft = cg->add_function<ConstantMinusX>({i_it}, 1.f);
    // write memory cell
    //VariableIndex i_awt;
    Expression i_awt;
    if (has_prev_state)
      i_awt = vars[BV] + vars[X2C] * in + vars[H2C]*i_h_tm1;
      //i_awt = cg->add_function<AffineTransform>({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
      i_awt = vars[BV] + vars[X2C] * in;
      //i_awt = cg->add_function<AffineTransform>({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);
    //VariableIndex i_wt = cg->add_function<Tanh>({i_awt});
    // output
    if (has_prev_state) {
      //VariableIndex i_nwt = cg->add_function<CwiseMultiply>({i_it, i_wt});
      Expression i_nwt = cwise_multiply(i_it,i_wt);
      //VariableIndex i_crt = cg->add_function<CwiseMultiply>({i_ft, i_c_tm1});
      Expression i_crt = cwise_multiply(i_ft,i_c_tm1);
      //ct[i] = cg->add_function<Sum>({i_crt, i_nwt}); // new memory cell at time t
      ct[i] = i_crt + i_nwt;
    } else {
      //ct[i] = cg->add_function<CwiseMultiply>({i_it, i_wt});
      ct[u] = cwise_multiply(i_it,i_wt);
    }
 
    //VariableIndex i_aot;
    Expression i_aot;
    if (has_prev_state)
      i_aot = vars[BO] + vars[X2O] * in + vars[H2O] * i_h_tm1 + vars[C2O] * ct[i];
      //i_aot = cg->add_function<AffineTransform>({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], ct[i]});
    else
      i_aot = vars[BO] + vars[X2O] * in;
      //i_aot = cg->add_function<AffineTransform>({vars[BO], vars[X2O], in});
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cwise_multiply(i,ot,ph_t);

    cerr << "Input_impl: Did all" << endl;
    //VariableIndex i_ot = cg->add_function<LogisticSigmoid>({i_aot});
    //VariableIndex ph_t = cg->add_function<Tanh>({ct[i]});
    //in = ht[i] = cg->add_function<CwiseMultiply>({i_ot, ph_t});
  }
  return ht.back();
}

} // namespace cnn
