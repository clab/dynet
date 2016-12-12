#include "dynet/rnn.h"
#include "dynet/io-macros.h"

#include <string>
#include <cassert>
#include <vector>
#include <fstream>
#include <iostream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>


#include "dynet/nodes.h"
#include "dynet/expr.h"

using namespace std;
using namespace dynet::expr;
using namespace dynet;

namespace dynet {

enum { X2H=0, H2H, HB, L2H };

RNNBuilder::~RNNBuilder() {}

void RNNBuilder::save_parameters_pretraining(const string& fname) const {
  cerr << "RNNBuilder::save_parameters_pretraining not overridden.\n";
  abort();
}

void RNNBuilder::load_parameters_pretraining(const string& fname) {
  cerr << "RNNBuilder::load_parameters_pretraining not overridden.\n";
  abort();
}

template<class Archive>
void RNNBuilder::serialize(Archive& ar, const unsigned int) {
  ar & cur;
  ar & head;
  ar & sm;
} 

DYNET_SERIALIZE_IMPL(RNNBuilder)

SimpleRNNBuilder::SimpleRNNBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model& model,
                       bool support_lags) : layers(layers), lagging(support_lags) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    Parameter p_x2h = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2h = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_hb = model.add_parameters({hidden_dim});
    vector<Parameter> ps = {p_x2h, p_h2h, p_hb};
    if (lagging)
        ps.push_back(model.add_parameters({hidden_dim, hidden_dim}));
    params.push_back(ps);
    layer_input_dim = hidden_dim;
  }
  dropout_rate = 0.f;
}

void SimpleRNNBuilder::new_graph_impl(ComputationGraph& cg) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    Parameter p_x2h = params[i][X2H];
    Parameter p_h2h = params[i][H2H];
    Parameter p_hb = params[i][HB];
    Expression i_x2h =  parameter(cg,p_x2h);
    Expression i_h2h =  parameter(cg,p_h2h);
    Expression i_hb =  parameter(cg,p_hb);
    vector<Expression> vars = {i_x2h, i_h2h, i_hb};

    if (lagging) {
        Parameter p_l2h = params[i][L2H];
        Expression i_l2h =  parameter(cg,p_l2h);
        vars.push_back(i_l2h);
    }

    param_vars.push_back(vars);
  }
}

void SimpleRNNBuilder::start_new_sequence_impl(const vector<Expression>& h_0) {
  h.clear();
  h0 = h_0;
  if (h0.size()) { assert(h0.size() == layers); }
}

Expression SimpleRNNBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  if (h_new.size()) { assert(h_new.size() == layers); }
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression y = h_new[i];
    h[t][i] = y;
  }
  return h[t].back();
}

Expression SimpleRNNBuilder::add_input_impl(int prev, const Expression &in) {
  if(dropout_rate != 0.f)
    throw std::runtime_error("SimpleRNNBuilder doesn't support dropout yet");
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));

  Expression x = in;

  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];

    // y <--- f(x)
    Expression y = affine_transform({vars[2], vars[0], x});

    // y <--- g(y_prev)
    if (prev == -1 && h0.size() > 0)
      y = affine_transform({y, vars[1], h0[i]});
    else if (prev >= 0)
      y = affine_transform({y, vars[1], h[prev][i]});

    // x <--- tanh(y)
    x = h[t][i] = tanh(y);
  }
  return h[t].back();
}

Expression SimpleRNNBuilder::add_auxiliary_input(const Expression &in, const Expression &aux) {
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));

  Expression x = in;

  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    assert(vars.size() >= L2H + 1);

    Expression y = affine_transform({vars[HB], vars[X2H], x, vars[L2H], aux});

    if (t == 0 && h0.size() > 0)
      y = affine_transform({y, vars[H2H], h0[i]});
    else if (t >= 1)
      y = affine_transform({y, vars[H2H], h[t-1][i]});

    x = h[t][i] = tanh(y);
  }
  return h[t].back();
}

void SimpleRNNBuilder::copy(const RNNBuilder & rnn) {
  const SimpleRNNBuilder & rnn_simple = (const SimpleRNNBuilder&)rnn;
  assert(params.size() == rnn_simple.params.size());
  for(size_t i = 0; i < rnn_simple.params.size(); ++i) {
      params[i][0] = rnn_simple.params[i][0];
      params[i][1] = rnn_simple.params[i][1];
      params[i][2] = rnn_simple.params[i][2];
  }
}

void SimpleRNNBuilder::save_parameters_pretraining(const string& fname) const {
  cerr << "Writing parameters to " << fname << endl;
  ofstream of(fname);
  assert(of);
  boost::archive::binary_oarchive oa(of);
  std::string id = "SimpleRNNBuilder:params";
  oa << id;
  oa << layers;
  for (unsigned i = 0; i < layers; ++i) {
    for (auto p : params[i]) {
      oa << p.get()->values;
    }
  }
}

void SimpleRNNBuilder::load_parameters_pretraining(const string& fname) {
  cerr << "Loading parameters from " << fname << endl;
  ifstream of(fname);
  assert(of);
  boost::archive::binary_iarchive ia(of);
  std::string id;
  ia >> id;
  if (id != "SimpleRNNBuilder:params") {
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
void SimpleRNNBuilder::serialize(Archive& ar, const unsigned int) {
  ar & boost::serialization::base_object<RNNBuilder>(*this);
  ar & params;
  ar & layers;
  ar & lagging;
}
DYNET_SERIALIZE_IMPL(SimpleRNNBuilder)

} // namespace dynet

BOOST_CLASS_EXPORT_IMPLEMENT(dynet::RNNBuilder)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::SimpleRNNBuilder)
