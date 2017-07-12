#define BOOST_TEST_MODULE TEST_RNN

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/rnn.h>
#include <dynet/lstm.h>
#include <dynet/fast-lstm.h>
#include <dynet/gru.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include "test.h"
#include <stdexcept>
#include <fstream>

using namespace dynet;
using namespace std;


struct RNNTest {
  RNNTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"RNNTest", "--dynet-seed", "10", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }
    seq_vals = {1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 0.f, 1.f, 1.f, 1.f, 0.f, 1.f};
    ones_vals = {1.f, 1.f, 1.f};
    param_vals = {1.1f, -2.2f, 3.3f};
    param2_vals = {1.1f, -2.2f, 3.3f};
  }
  ~RNNTest() {
    // This was causing double deallocation errors?
    // for (auto x : av) free(x);
  }

  template <class T>
  std::string print_vec(const std::vector<T> vec) {
    ostringstream oss;
    if (vec.size()) oss << vec[0];
    for (size_t i = 1; i < vec.size(); i++)
      oss << ' ' << vec[i];
    return oss.str();
  }

  std::vector<float> ones_vals, param_vals, param2_vals, seq_vals;
  std::vector<char*> av;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(rnn_test, RNNTest);

#define DYNET_RNN_GRADIENT_TEST_CASE(name, RNN_TYPE)      \
BOOST_AUTO_TEST_CASE( name ) {                            \
  dynet::ParameterCollection mod;                         \
  RNN_TYPE rnn(2,3,4,mod);                                \
  dynet::ComputationGraph cg;                             \
  rnn.new_graph(cg);                                      \
  rnn.start_new_sequence();                               \
  for(unsigned i=0;i<4;i++){                              \
    Expression x = dynet::input(cg,Dim({3}), ones_vals);  \
    rnn.add_input(x);                                     \
  }                                                       \
  Expression z = squared_norm(rnn.final_h()[1]);          \
  BOOST_CHECK(check_grad(mod, z, 0));                     \
}                                                         \

DYNET_RNN_GRADIENT_TEST_CASE(simple_rnn_gradient, dynet::SimpleRNNBuilder)

DYNET_RNN_GRADIENT_TEST_CASE(vanilla_lstm_gradient, dynet::VanillaLSTMBuilder)

DYNET_RNN_GRADIENT_TEST_CASE(lstm_gradient, dynet::LSTMBuilder)

DYNET_RNN_GRADIENT_TEST_CASE(gru_gradient, dynet::GRUBuilder)

DYNET_RNN_GRADIENT_TEST_CASE(fast_lstm, dynet::FastLSTMBuilder)

BOOST_AUTO_TEST_CASE( vanilla_lstm_ln_gradient ) {
  dynet::ParameterCollection mod;
  dynet::VanillaLSTMBuilder vanilla_lstm(2, 3, 10, mod, true);
  dynet::ComputationGraph cg;
  vanilla_lstm.new_graph(cg);
  vanilla_lstm.start_new_sequence();
  for (unsigned i = 0; i < 4; i++) {
    Expression x = dynet::input(cg, Dim({3}), ones_vals);
    vanilla_lstm.add_input(x);
  }
  Expression z = squared_norm(vanilla_lstm.final_h()[1]);
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( lstm_node_forward ) {
  cout << "z\n";
  dynet::ParameterCollection mod;
  unsigned input_dim = 3;
  unsigned hidden_dim = 5;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);
  vanilla_lstm_builder.start_new_sequence();

  cout << "a\n";
  enum { _X2I, _H2I, _BI, _X2F, _H2F, _BF, _X2O, _H2O, _BO, _X2G, _H2G, _BG };
  cout << "b\n";

  Expression Wx_i = parameter(cg, vanilla_lstm_builder.params[0][_X2I]);
  cout << vanilla_lstm_builder.params[0].size() << " - c\n";
  Expression Wx_f = parameter(cg, vanilla_lstm_builder.params[0][_X2F]);
  cout << "d\n";
  Expression Wx_o = parameter(cg, vanilla_lstm_builder.params[0][_X2O]);
      cout << "e\n";
  Expression Wx_g = parameter(cg, vanilla_lstm_builder.params[0][_X2G]);
    cout << "f\n";

  Expression Wh_i = parameter(cg, vanilla_lstm_builder.params[0][_H2I]);
  Expression Wh_f = parameter(cg, vanilla_lstm_builder.params[0][_H2F]);
  Expression Wh_o = parameter(cg, vanilla_lstm_builder.params[0][_H2O]);
  Expression Wh_g = parameter(cg, vanilla_lstm_builder.params[0][_H2G]);

  Expression b_i = parameter(cg, vanilla_lstm_builder.params[0][_BI]);
  Expression b_f = parameter(cg, vanilla_lstm_builder.params[0][_BF]);
  Expression b_o = parameter(cg, vanilla_lstm_builder.params[0][_BO]);
  Expression b_g = parameter(cg, vanilla_lstm_builder.params[0][_BG]);

    cout << "g\n";

  Expression hc_tm1 = dynet::input(cg, Dim({hidden_dim*2}), {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
  for (unsigned i = 0; i < 1; i++) { // TODO: more inputs
    Expression x = dynet::input(cg, Dim({input_dim}), ones_vals);
    vanilla_lstm_builder.add_input(x);
    Expression hc_t = dynet::vanilla_lstm(x, hc_tm1, Wx_i, Wx_f, Wx_o, Wx_g, Wh_i, Wh_f, Wh_o, Wh_g, b_i, b_f, b_o, b_g);
    cout << "h\n";
  }
//  BOOST_CHECK(check_grad(mod, z, 0));
}


BOOST_AUTO_TEST_SUITE_END()
