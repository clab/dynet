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
using namespace dynet::expr;
using namespace std;


struct RNNTest {
  RNNTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"RNNTest", "--dynet-seed", "10", "--dynet-autobatch", "1", "--dynet-mem", "10"}) {
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
  dynet::Model mod;                                       \
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
  dynet::Model mod;
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


BOOST_AUTO_TEST_SUITE_END()
