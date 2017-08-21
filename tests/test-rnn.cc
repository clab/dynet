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
    param_1_vals = {1.1f};
    param_2_vals = {-2.2f, 3.3f};
    param_3_vals = {0.f, 0.1f, 0.2f};
    param_4_vals = {1.f, 0.1f, 1.2f, -0.3f};
    param_5_zeros_vals = {0.0, 0.0, 0.0, 0.0, 0.0};
    param_6_vals = {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    param_8_vals = {0.1f, 0.1f, 0.2f, 0.3f, 0.2f, 0.3f, 0.4f, 0.5f};
    param_10_vals = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.3f, 0.4f, 0.5f};
    param_10_zeros_vals = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param_12_vals = {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, 0.4f, 0.5f, 0.3f, 0.4f, 0.5f};
    param_24_vals = {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    param_36_vals = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, -0.5};
    param_60_vals = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.3f, 0.4f, 0.5f};
    param_6_mask_vals = {0.f, 1.0f, 1.0f, 1.f, 1.0f, 0.0f};
    param_8_mask_vals = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0};
    param_10_mask_vals = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    param_1 = mod.add_parameters(Dim({1}));
    TensorTools::set_elements(param_1.get_storage().values, param_1_vals);
    param_2 = mod.add_parameters(Dim({2}));
    TensorTools::set_elements(param_2.get_storage().values, param_2_vals);
    param_3 = mod.add_parameters(Dim({3}));
    TensorTools::set_elements(param_3.get_storage().values, param_3_vals);
    param_4 = mod.add_parameters(Dim({4}));
    TensorTools::set_elements(param_4.get_storage().values, param_4_vals);
    param_5_zeros = mod.add_parameters(Dim({5}));
    TensorTools::set_elements(param_5_zeros.get_storage().values, param_5_zeros_vals);
    param_6 = mod.add_parameters(Dim({6}));
    TensorTools::set_elements(param_6.get_storage().values, param_6_vals);
    param_8 = mod.add_parameters(Dim({8}));
    TensorTools::set_elements(param_8.get_storage().values, param_8_vals);
    param_10 = mod.add_parameters(Dim({10}));
    TensorTools::set_elements(param_10.get_storage().values, param_10_vals);
    param_10_zeros = mod.add_parameters(Dim({10}));
    TensorTools::set_elements(param_10_zeros.get_storage().values, param_10_zeros_vals);
    param_12 = mod.add_parameters(Dim({12}));
    TensorTools::set_elements(param_6.get_storage().values, param_12_vals);
    param_24 = mod.add_parameters(Dim({24}));
    TensorTools::set_elements(param_24.get_storage().values, param_24_vals);
    param_36 = mod.add_parameters(Dim({36}));
    TensorTools::set_elements(param_36.get_storage().values, param_36_vals);
    param_60 = mod.add_parameters(Dim({60}));
    TensorTools::set_elements(param_60.get_storage().values, param_60_vals);
    param_6_mask = mod.add_parameters(Dim({6}));
    TensorTools::set_elements(param_6_mask.get_storage().values, param_6_mask_vals);
    param_8_mask = mod.add_parameters(Dim({8}));
    TensorTools::set_elements(param_8_mask.get_storage().values, param_8_mask_vals);
    param_10_mask = mod.add_parameters(Dim({10}));
    TensorTools::set_elements(param_10_mask.get_storage().values, param_10_mask_vals);

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

  std::vector<float> ones_vals, param_vals, param_1_vals, param_2_vals, seq_vals, param_3_vals, param_4_vals, param_5_zeros_vals, param_6_vals, param_8_vals, param_10_vals, param_10_zeros_vals, param_12_vals, param_24_vals, param_36_vals, param_60_vals, param_6_mask_vals, param_8_mask_vals, param_10_mask_vals;
  std::vector<char*> av;
  dynet::ParameterCollection mod;
  dynet::Parameter param_1, param_2, param_3, param_4, param_5_zeros, param_6, param_8, param_10, param_10_zeros, param_12, param_24, param_36, param_60, param_6_mask, param_8_mask, param_10_mask;
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

DYNET_RNN_GRADIENT_TEST_CASE(compact_vanilla_lstm_gradient, dynet::CompactVanillaLSTMBuilder)

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

BOOST_AUTO_TEST_CASE( lstm_node_fwd ) {
  dynet::ParameterCollection mod;
  unsigned input_dim = 3;
  unsigned hidden_dim = 5;
  unsigned batch_size = 1;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);
  vanilla_lstm_builder.start_new_sequence({dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.f, 0.f, 0.f, 0.f}),
					   dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.f, 0.f, 0.f, 0.f})});

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]);

  Expression c_tm1 = -dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.f, 0.f, 0.f, 0.f});
  Expression h_tm1 = -dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.f, 0.f, 0.f, 0.f});
  Expression x = -dynet::input(cg, Dim({input_dim}, batch_size), {1.f, 1.f, 1.f});
  for (unsigned i = 0; i < 3; i++) {
    const Tensor& builder_h = vanilla_lstm_builder.add_input(x).value();
    Expression gates_t = dynet::vanilla_lstm_gates(x, h_tm1, Wx, Wh, b);
    Expression c_t = dynet::vanilla_lstm_c(c_tm1, gates_t);
    Expression h_t = dynet::vanilla_lstm_h(c_t, gates_t);
    for(unsigned i=0; i < hidden_dim; i++){
      BOOST_CHECK_CLOSE(as_vector(h_t.value())[i], as_vector(builder_h)[i], 0.001);
    }
    c_tm1 = c_t;
    h_tm1 = h_t;
  }
}

BOOST_AUTO_TEST_CASE( lstm_node_bwd ) {
  dynet::ParameterCollection mod;
  unsigned input_dim = 3;
  unsigned hidden_dim = 5;
  unsigned batch_size = 1;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]);

  Expression c_tm1 = dynet::reshape(dynet::parameter(cg, param_5_zeros), Dim({hidden_dim},batch_size));
  Expression h_tm1 = dynet::reshape(dynet::parameter(cg, param_5_zeros), Dim({hidden_dim},batch_size));
  Expression x = dynet::reshape(dynet::parameter(cg, param_3), Dim({input_dim},batch_size));
  for (unsigned i = 0; i < 3; i++) {
    Expression gates_t = dynet::vanilla_lstm_gates(x, h_tm1, Wx, Wh, b);
    Expression c_t = dynet::vanilla_lstm_c(c_tm1, gates_t);
    Expression h_t = dynet::vanilla_lstm_h(c_t, gates_t);
    c_tm1 = c_t;
    h_tm1 = h_t;
  }
  Expression z = squared_norm(sum_batches(h_tm1 + c_tm1));
  BOOST_CHECK(check_grad(mod, z, 0));
}


BOOST_AUTO_TEST_CASE( lstm_node_batched_forward ) {
  dynet::ParameterCollection mod;
  unsigned input_dim = 3;
  unsigned hidden_dim = 5;
  unsigned batch_size = 2;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);
  vanilla_lstm_builder.start_new_sequence({dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f}),
					   dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f})});

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]);

  Expression c_tm1 = dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});
  Expression h_tm1 = dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});
  Expression x = dynet::input(cg, Dim({input_dim}, batch_size), {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
  for (unsigned i = 0; i < 3; i++) {
    const Tensor& builder_h = vanilla_lstm_builder.add_input(x).value();
    Expression gates_t = dynet::vanilla_lstm_gates(x, h_tm1, Wx, Wh, b);
    Expression c_t = dynet::vanilla_lstm_c(c_tm1, gates_t);
    Expression h_t = dynet::vanilla_lstm_h(c_t, gates_t);
    for(unsigned i=0; i < hidden_dim; i++){
      BOOST_CHECK_CLOSE(as_vector(h_t.value())[i], as_vector(builder_h)[i], 0.001);
    }
    c_tm1 = c_t;
    h_tm1 = h_t;
  }
}

BOOST_AUTO_TEST_CASE( lstm_node_h_fwd ) {
  unsigned hidden_dim = 3;
  unsigned batch_size = 2;
  dynet::ComputationGraph cg;

  Expression c_t = dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.7f});
  Expression gates_t = dynet::input(cg, Dim({hidden_dim*4}, batch_size), {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.f, -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, 0.01f, 0.11f, 0.21f, 0.31f, 0.41f, 0.51f, -0.01f, -0.11f, -0.21f, -0.31f, -0.41f, -0.51f});
  Expression h_t = vanilla_lstm_h(c_t, gates_t);
  BOOST_CHECK_SMALL(as_vector(pick_batch_elem(h_t, (unsigned)0).value())[0], (float)1.0e-6);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(h_t, (unsigned)0).value())[1], -0.009966799462, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(h_t, (unsigned)0).value())[2], -0.03947506404, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(h_t, (unsigned)1).value())[0], -0.002913126125, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(h_t, (unsigned)1).value())[1], -0.04179438585, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(h_t, (unsigned)1).value())[2], -0.1269172332, 0.001);
}

BOOST_AUTO_TEST_CASE( lstm_node_h_gradient ) {
  unsigned hidden_dim = 3;
  unsigned batch_size = 2;
  dynet::ComputationGraph cg;

  Expression c_t = dynet::reshape(dynet::parameter(cg, param_6), Dim({hidden_dim},batch_size));
  Expression gates_t = dynet::reshape(dynet::parameter(cg, param_24), Dim({hidden_dim*4},batch_size));
  Expression h_t = vanilla_lstm_h(c_t, gates_t);
  Expression z = squared_norm(sum_batches(h_t));
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( lstm_node_c_gradient ) {
  unsigned hidden_dim = 3;
  unsigned batch_size = 2;
  dynet::ComputationGraph cg;

  Expression c_tm1 = dynet::reshape(dynet::parameter(cg, param_6), Dim({hidden_dim},batch_size));
  Expression gates_t = dynet::reshape(dynet::parameter(cg, param_24), Dim({hidden_dim*4},batch_size));
  Expression c_t = vanilla_lstm_c(c_tm1, gates_t);
  Expression z = sum_elems(sum_batches(c_t));
  z.value();
  BOOST_CHECK(check_grad(mod, z, 0));
}
BOOST_AUTO_TEST_CASE( lstm_node_c_fwd ) {
  unsigned hidden_dim = 3;
  unsigned batch_size = 2;
  dynet::ComputationGraph cg;

  Expression c_tm1 = dynet::input(cg, Dim({hidden_dim}, batch_size), {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f});
  Expression gates_t = dynet::input(cg, Dim({hidden_dim*4}, batch_size), {0.f, 0.1f, 0.2f,         0.3f, 0.4f, 0.5f,        0.f, -0.1f, -0.2f,        -0.3f, -0.4f, -0.5f,
									  0.01f, 0.11f, 0.21f,     0.31f, 0.41f, 0.51f,    -0.01f, -0.11f, -0.21f,    -0.31f, -0.41f, -0.51f});
  Expression c_t = vanilla_lstm_c(c_tm1, gates_t);
  BOOST_CHECK_SMALL(as_vector(pick_batch_elem(c_t, (unsigned)0).value())[0], (float)1.0e-6);
  BOOST_CHECK_SMALL(as_vector(pick_batch_elem(c_t, (unsigned)0).value())[1], (float)1.0e-6);
  BOOST_CHECK_SMALL(as_vector(pick_batch_elem(c_t, (unsigned)0).value())[2], (float)1.0e-6);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(c_t, (unsigned)1).value())[0], 0.0899, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(c_t, (unsigned)1).value())[1], 0.1189, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(c_t, (unsigned)1).value())[2], 0.1479, 0.001);
}
BOOST_AUTO_TEST_CASE( lstm_node_gates_fwd ) {
  unsigned input_dim = 2;
  unsigned hidden_dim = 2;
  unsigned batch_size = 2;
  dynet::ComputationGraph cg;

  Expression x_t = dynet::input(cg, Dim({input_dim}, batch_size), {0.f, 0.1f, 0.2f, 0.3f});
//    cout << "x_t: " << print_vec(as_vector(x_t.value())) << "\n";
//    cout << "x_t b0: " << print_vec(as_vector(pick_batch_elem(x_t, (unsigned)0).value())) << "\n";
//    cout << "x_t b1: " << print_vec(as_vector(pick_batch_elem(x_t, (unsigned)1).value())) << "\n";
  Expression h_tm1 = dynet::input(cg, Dim({input_dim}, batch_size), {0.f, -0.1f, 0.2f, -0.3f});
  Expression Wx = dynet::input(cg, Dim({hidden_dim*4, input_dim}, 1), {0.f, 1.1f, 2.2f, 3.3f, 0.f, 1.1f, 2.2f, 3.3f, 0.f, 1.1f, 2.2f, 3.3f, 0.f, 1.1f, 2.2f, 3.3f});
  Expression Wh = dynet::input(cg, Dim({hidden_dim*4, hidden_dim}, 1), {0.1f, 1.2f, 2.3f, 3.4f, 0.1f, 1.2f, 2.3f, 3.4f, 0.1f, 1.2f, 2.3f, 3.4f, 0.1f, 1.2f, 2.3f, 3.4f});
  Expression b = dynet::input(cg, Dim({hidden_dim*4}, 1), {0.f, 0.1f, 0.f, 0.1f, 0.f, 0.1f, 0.f, 0.1f});
  Expression gates = vanilla_lstm_gates(x_t, h_tm1, Wx, Wh, b);
  // Wx*x = 0.11 0.33 / 0.33 1.43
  // Wh*h = -0.12 -0.34 / -0.34 -0.56
  // + b  = -0.01 0.09 / -0.01 0.97
  // intermediate step:
  // Wx*x + Wh*h + b =
  // [-0.01 0.09 0.99 1.09 -0.01 0.09 -0.01 0.09]^T  (including forget bias + 1)
  // [-0.01 0.53 1.87 2.41 -0.01 0.53 0.87 1.41]^T   (including forget bias + 1)
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[0], 0.497500021, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[1], 0.522484825, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[2], 0.729087922, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[3], 0.748381722, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[4], 0.497500021, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[5], 0.522484825, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[6], -0.009999656681, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[7], 0.08975778475, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[0], 0.497500021, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[1], 0.629483112, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[2], 0.866458277, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[3], 0.917586682, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[4], 0.497500021, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[5], 0.629483112, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[6], 0.7013741309, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[7], 0.8874941329, 0.001);
}

BOOST_AUTO_TEST_CASE( lstm_node_gates_bwd ) {
  unsigned input_dim = 5;
  unsigned hidden_dim = 3;
  unsigned batch_size = 2;
  dynet::ComputationGraph cg;

  Expression x_t = dynet::reshape(dynet::parameter(cg, param_10), Dim({input_dim},batch_size));
  Expression h_tm1 = dynet::reshape(dynet::parameter(cg, param_6), Dim({hidden_dim},batch_size));
  Expression Wx = dynet::reshape(dynet::parameter(cg, param_60), Dim({hidden_dim*4, input_dim},1));
  Expression Wh = dynet::reshape(dynet::parameter(cg, param_36), Dim({hidden_dim*4, hidden_dim},1));
  Expression b = dynet::reshape(dynet::parameter(cg, param_12), Dim({hidden_dim*4},1));

  Expression gates = vanilla_lstm_gates(x_t, h_tm1, Wx, Wh, b);

  Expression z = squared_norm(sum_batches(gates));
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( lstm_node_gates_dropout_fwd ) {
  unsigned input_dim = 2;
  unsigned hidden_dim = 2;
  unsigned batch_size = 2;
  dynet::ComputationGraph cg;

  Expression x_t = dynet::input(cg, Dim({input_dim}, batch_size), {0.0, 0.1, 0.2, 0.3});
  Expression mask_x = dynet::input(cg, Dim({input_dim}, batch_size), {0.0, 1, 1, 1});
//    cout << "x_t: " << print_vec(as_vector(x_t.value())) << "\n";
//    cout << "x_t b0: " << print_vec(as_vector(pick_batch_elem(x_t, (unsigned)0).value())) << "\n";
//    cout << "x_t b1: " << print_vec(as_vector(pick_batch_elem(x_t, (unsigned)1).value())) << "\n";
  Expression h_tm1 = dynet::input(cg, Dim({input_dim}, batch_size), {0.f, -0.1, 0.2, -0.3});
  Expression mask_h = dynet::input(cg, Dim({input_dim}, batch_size), {0.0, 0.0, 1.0, 1.0});
  Expression Wx = dynet::input(cg, Dim({hidden_dim*4, input_dim}, 1), {0.f, 1.1f, 2.2f, 3.3f, 0.f, 1.1f, 2.2f, 3.3f, 0.f, 1.1f, 2.2f, 3.3f, 0.f, 1.1f, 2.2f, 3.3f});
  Expression Wh = dynet::input(cg, Dim({hidden_dim*4, hidden_dim}, 1), {0.1f, 1.2f, 2.3f, 3.4f, 0.1f, 1.2f, 2.3f, 3.4f, 0.1f, 1.2f, 2.3f, 3.4f, 0.1f, 1.2f, 2.3f, 3.4f});
  Expression b = dynet::input(cg, Dim({hidden_dim*4}, 1), {0.f, 0.1f, 0.f, 0.1f, 0.f, 0.1f, 0.f, 0.1f});
  Expression gates = vanilla_lstm_gates_dropout(x_t, h_tm1, Wx, Wh, b, mask_x, mask_h);
  // Wx*x = 0.11 0.33 / 0.33 1.43
  // Wh*h = -0.12 -0.34 / -0.34 -0.56
  // + b  = -0.01 0.09 / -0.01 0.97
  // intermediate step:
  // Wx*x + Wh*h + b =
  // [0 0.21 0.22 0.43 1 1.21 0.22 0.43]^T  (including forget bias + 1)
  // [-0.01 0.53 1.87 2.41 -0.01 0.53 0.87 1.41]^T   (including forget bias + 1)
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[0], 0.5, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[1], 0.5523079096, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[2], 0.7720635494, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[3], 0.8069013158, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[4], 0.5, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[5], 0.5523079096, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[6], 0.2165180615, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)0).value())[7], 0.4053213087, 0.001);

  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[0], 0.497500021, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[1], 0.629483112, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[2], 0.866458277, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[3], 0.917586682, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[4], 0.497500021, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[5], 0.629483112, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[6], 0.7013741309, 0.001);
  BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates, (unsigned)1).value())[7], 0.8874941329, 0.001);
}

BOOST_AUTO_TEST_CASE( lstm_node_dropout_bwd ) {
  dynet::ParameterCollection mod;
  unsigned input_dim = 3;
  unsigned hidden_dim = 5;
  unsigned batch_size = 2;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]) + 0.5;

  Expression c_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));
  Expression h_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));
  Expression x = dynet::reshape(dynet::parameter(cg, param_6), Dim({input_dim},batch_size));

  Expression mask_x = dynet::reshape(dynet::parameter(cg, param_6_mask), Dim({input_dim},batch_size))*0.9;
  Expression mask_h = dynet::reshape(dynet::parameter(cg, param_10_mask), Dim({hidden_dim},batch_size))*0.8;

  for (unsigned i = 0; i < 3; i++) {
    Expression gates_t = dynet::vanilla_lstm_gates_dropout(x, h_tm1, Wx, Wh, b, mask_x, mask_h, 0.0f);
    Expression c_t = dynet::vanilla_lstm_c(c_tm1, gates_t);
    Expression h_t = dynet::vanilla_lstm_h(c_t, gates_t);
    c_tm1 = c_t;
    h_tm1 = h_t;
  }
  Expression z = squared_norm(sum_batches(h_tm1 + c_tm1));
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( lstm_node_multi_input_bwd ) {
  dynet::ParameterCollection mod;
  unsigned input_dim = 4;
  unsigned hidden_dim = 5;
  unsigned batch_size = 2;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]) + 0.5;

  Expression c_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));
  Expression h_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));
  Expression x1 = dynet::reshape(dynet::parameter(cg, param_2), Dim({1},batch_size));
  Expression x3 = dynet::reshape(dynet::parameter(cg, param_6), Dim({3},batch_size));

  for (unsigned i = 0; i < 3; i++) {
    Expression gates_t = dynet::vanilla_lstm_gates_concat({x1, x3}, h_tm1, Wx, Wh, b, 0.0f);
    Expression c_t = dynet::vanilla_lstm_c(c_tm1, gates_t);
    Expression h_t = dynet::vanilla_lstm_h(c_t, gates_t);
    c_tm1 = c_t;
    h_tm1 = h_t;
  }
  Expression z = squared_norm(sum_batches(h_tm1 + c_tm1));
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( lstm_node_multi_input_fwd ) {
  dynet::ParameterCollection mod;
  unsigned input_dim = 4;
  unsigned hidden_dim = 5;
  unsigned batch_size = 2;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]) + 0.5;

  Expression c_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));
  Expression h_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));

  Expression x = dynet::reshape(dynet::parameter(cg, param_8), Dim({input_dim},batch_size));
  Expression x1 = dynet::pick_range(x, 0, 1);
  Expression x3 = dynet::pick_range(x, 1, 4);

  for (unsigned i = 0; i < 3; i++) {
    Expression gates_t = dynet::vanilla_lstm_gates_concat({x1, x3}, h_tm1, Wx, Wh, b, 0.0f);
    Expression gates_t_2 = dynet::vanilla_lstm_gates_concat({x}, h_tm1, Wx, Wh, b, 0.0f);
    for(int b=0; b<batch_size; b++){
      for(int u=0; u<4*hidden_dim; u++){
        BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates_t, (unsigned)b).value())[u],
        as_vector(pick_batch_elem(gates_t_2, (unsigned)b).value())[u],
        0.001);
      }
    }
    Expression c_t = dynet::vanilla_lstm_c(c_tm1, gates_t);
    Expression h_t = dynet::vanilla_lstm_h(c_t, gates_t);
    c_tm1 = c_t;
    h_tm1 = h_t;
  }
}

BOOST_AUTO_TEST_CASE( lstm_node_dropout_multi_input_bwd ) {
  dynet::ParameterCollection mod;
  unsigned input_dim = 4;
  unsigned hidden_dim = 5;
  unsigned batch_size = 2;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]) + 0.5;

  Expression c_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));
  Expression h_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));
  Expression x1 = dynet::reshape(dynet::parameter(cg, param_2), Dim({1},batch_size));
  Expression x3 = dynet::reshape(dynet::parameter(cg, param_6), Dim({3},batch_size));

  Expression mask_x = dynet::reshape(dynet::parameter(cg, param_8_mask), Dim({input_dim},batch_size))*0.9;
  Expression mask_h = dynet::reshape(dynet::parameter(cg, param_10_mask), Dim({hidden_dim},batch_size))*0.8;

  for (unsigned i = 0; i < 3; i++) {
    Expression gates_t = dynet::vanilla_lstm_gates_dropout_concat({x1, x3}, h_tm1, Wx, Wh, b, mask_x, mask_h, 0.0f);
    Expression c_t = dynet::vanilla_lstm_c(c_tm1, gates_t);
    Expression h_t = dynet::vanilla_lstm_h(c_t, gates_t);
    c_tm1 = c_t;
    h_tm1 = h_t;
  }
  Expression z = squared_norm(sum_batches(h_tm1 + c_tm1));
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( lstm_node_dropout_multi_input_fwd ) {
  dynet::ParameterCollection mod;
  unsigned input_dim = 4;
  unsigned hidden_dim = 5;
  unsigned batch_size = 2;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]) + 0.5;

  Expression c_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));
  Expression h_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));

  Expression x = dynet::reshape(dynet::parameter(cg, param_8), Dim({input_dim},batch_size));
  Expression x1 = dynet::pick_range(x, 0, 1);
  Expression x3 = dynet::pick_range(x, 1, 4);

  Expression mask_x = dynet::reshape(dynet::parameter(cg, param_8_mask), Dim({input_dim},batch_size))*0.9;
  Expression mask_h = dynet::reshape(dynet::parameter(cg, param_10_mask), Dim({hidden_dim},batch_size))*0.8;

  for (unsigned i = 0; i < 3; i++) {
    Expression gates_t = dynet::vanilla_lstm_gates_dropout_concat({x1, x3}, h_tm1, Wx, Wh, b, mask_x, mask_h, 0.0f);
    Expression gates_t_2 = dynet::vanilla_lstm_gates_dropout(x, h_tm1, Wx, Wh, b, mask_x, mask_h, 0.0f);
    for(int b=0; b<batch_size; b++){
      for(int u=0; u<4*hidden_dim; u++){
        BOOST_CHECK_CLOSE(as_vector(pick_batch_elem(gates_t, (unsigned)b).value())[u],
        as_vector(pick_batch_elem(gates_t_2, (unsigned)b).value())[u],
        0.001);
      }
    }
    Expression c_t = dynet::vanilla_lstm_c(c_tm1, gates_t);
    Expression h_t = dynet::vanilla_lstm_h(c_t, gates_t);
    c_tm1 = c_t;
    h_tm1 = h_t;
  }
}

BOOST_AUTO_TEST_CASE( lstm_node_multi_weightnoise_runs ) {
  dynet::ParameterCollection mod;
  unsigned input_dim = 4;
  unsigned hidden_dim = 5;
  unsigned batch_size = 2;
  dynet::VanillaLSTMBuilder vanilla_lstm_builder(1, input_dim, hidden_dim, mod, false);
  dynet::ComputationGraph cg;
  vanilla_lstm_builder.new_graph(cg);

  Expression Wx = parameter(cg, vanilla_lstm_builder.params[0][0]);
  Expression Wh = parameter(cg, vanilla_lstm_builder.params[0][1]);
  Expression b = parameter(cg, vanilla_lstm_builder.params[0][2]) + 0.5;

  Expression c_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));
  Expression h_tm1 = dynet::reshape(dynet::parameter(cg, param_10_zeros), Dim({hidden_dim},batch_size));

  Expression x = dynet::reshape(dynet::parameter(cg, param_8), Dim({input_dim},batch_size));
  Expression x1 = dynet::pick_range(x, 0, 1);
  Expression x3 = dynet::pick_range(x, 1, 4);

  for (unsigned i = 0; i < 3; i++) {
    Expression gates_t = dynet::vanilla_lstm_gates_concat({x1, x3}, h_tm1, Wx, Wh, b, 0.1f);
    gates_t = dynet::vanilla_lstm_gates(x, h_tm1, Wx, Wh, b, 0.1f);
    Expression c_t = dynet::vanilla_lstm_c(c_tm1, gates_t);
    Expression h_t = dynet::vanilla_lstm_h(c_t, gates_t);
    c_tm1 = c_t;
    h_tm1 = h_t;
  }
}

BOOST_AUTO_TEST_SUITE_END()
