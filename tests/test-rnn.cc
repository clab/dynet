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
  dynet::ParameterCollection mod;
  unsigned input_dim = 3;
  unsigned hidden_dim = 5;
  unsigned batch_size = 2;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);
  vanilla_lstm_builder.start_new_sequence({dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}),
					   dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f})});

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]);

  Expression hc_tm1 = dynet::input(cg, Dim({hidden_dim*2}, batch_size), {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
  Expression x = dynet::input(cg, Dim({input_dim}, batch_size), {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
  for (unsigned i = 0; i < 2; i++) {
    const Tensor& builder_h = vanilla_lstm_builder.add_input(x).value();
    Expression hc_t = dynet::vanilla_lstm(x, hc_tm1, Wx, Wh, b);
    const Tensor& lstm_node_hc = hc_t.value();
    Tensor lstm_node_h(Dim({hidden_dim}, batch_size), lstm_node_hc.v, lstm_node_hc.device, DeviceMempool::FXS);
    Tensor lstm_node_c(Dim({hidden_dim}, batch_size), lstm_node_hc.v + hidden_dim * batch_size, lstm_node_hc.device, DeviceMempool::FXS);
    for(unsigned i=0; i < hidden_dim; i++){
      BOOST_CHECK_CLOSE(as_vector(lstm_node_h)[i], as_vector(builder_h)[i], 0.001);
    }
    hc_tm1 = hc_t;
  }
//  BOOST_CHECK(check_grad(mod, z, 0));
}


BOOST_AUTO_TEST_SUITE_END()
