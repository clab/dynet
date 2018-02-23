#define BOOST_TEST_MODULE TEST_RNN

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/exec.h>
#include <dynet/lstm.h>
#include <dynet/fast-lstm.h>
#include <dynet/gru.h>
#include <dynet/grad-check.h>
#include <dynet/param-init.h>
#include <boost/test/unit_test.hpp>
#include "test.h"
#include <stdexcept>
#include <fstream>

using namespace dynet;
using namespace std;


struct ExecTest {
  ExecTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"ExecTest", "--dynet-seed", "10", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      ADD_EXTRA_ARGUMENTS(av)
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }
  }
  ~ExecTest() {
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

  std::vector<char*> av;

};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(exec_test, ExecTest);

BOOST_AUTO_TEST_CASE( autobatch_lstm_gradient ) {
  vector<float> results;
  dynet::ParameterCollection mod;
  dynet::VanillaLSTMBuilder lstm(2, 3, 10, mod);
  dynet::LookupParameter lp = mod.add_lookup_parameters(10, {3});
  for(size_t i = 0; i < 3; ++i) {
    dynet::autobatch_flag = i;
    dynet::ComputationGraph cg;
    lstm.new_graph(cg);
    vector<Expression> losses;
    for(size_t j = 0; j < 3; ++j) {
      lstm.start_new_sequence();
      for(size_t k = 0; k < 3; ++k) {
        Expression x = dynet::lookup(cg, lp, j*3 + k);
        lstm.add_input(x);
      }
      losses.push_back(squared_norm(lstm.final_h()[1]));
    }
    losses.push_back(losses[0] + losses[2]);
    Expression z = dynet::sum(losses);
    results.push_back(as_scalar(z.value()));
    BOOST_CHECK(check_grad(mod, z, 0));
  }
  for(size_t i = 1; i < results.size(); ++i)
    BOOST_CHECK_CLOSE(results[0], results[i], 0.0001);
}

// TODO: This is commented out because it inexplicably causes problems only when
//       performing manual install on mac on Travis CI, despite the fact that it
//       works in my local mac environment. Until it becomes possible to debug
//       on Travis, it will remain commented out.
// BOOST_AUTO_TEST_CASE( autobatch_big_sum ) {
//   dynet::ParameterCollection mod;
//   vector<float> f1(50), f2(30);
//   for(size_t i = 0; i < 50; ++i) f1[i] = i / 5000.0;
//   for(size_t i = 0; i < 30; ++i) f2[i] = i / 3000.0;
//   dynet::LookupParameter lp1 = mod.add_lookup_parameters(10, {5}, dynet::ParameterInitFromVector(f1));
//   dynet::LookupParameter lp2 = mod.add_lookup_parameters(10, {3}, dynet::ParameterInitFromVector(f2));
//   vector<float> results;
//   for(size_t i = 0; i < 3; ++i) {
//     dynet::autobatch_flag = i;
//     dynet::ComputationGraph cg;
//     vector<Expression> v1, v2;
//     for(size_t j = 0; j < 10000; ++j) {
//       v1.push_back(dynet::lookup(cg, lp1, (j*3 % 10)));
//       v2.push_back(dynet::lookup(cg, lp2, (j*3 % 10)));   
//     }
//     Expression z = dynet::sum_elems(dynet::sum(v1)) + dynet::sum_elems(dynet::sum(v2));
//     results.push_back(as_scalar(z.value()));
//     BOOST_CHECK(check_grad(mod, z, 0));
//   }
//   for(size_t i = 1; i < results.size(); ++i)
//     BOOST_CHECK_CLOSE(results[0], results[i], 0.0001);
// }

BOOST_AUTO_TEST_CASE( param_after_node ) {
  auto autobatch_cache = dynet::autobatch_flag;
  for(size_t i = 0; i < 3; ++i) {
    dynet::autobatch_flag = i;
	  ComputationGraph cg;
	  ParameterCollection model;
	  Parameter param = model.add_parameters({ 1 });

	  Expression loss = zeroes(cg, { 1 });
	  parameter(cg, param);

	  cg.incremental_forward(loss);
	  cg.backward(loss);
  }
  dynet::autobatch_flag = autobatch_cache;
}

BOOST_AUTO_TEST_CASE( param_after_node_2 ) {
  auto autobatch_cache = dynet::autobatch_flag;
  for(size_t i = 0; i < 3; ++i) {
    dynet::autobatch_flag = i;
	  ComputationGraph cg;
	  ParameterCollection model;
	  LookupParameter param = model.add_lookup_parameters(10, { 1 });

	  lookup(cg, param, 1);
	  Expression loss = zeroes(cg, { 1 });

	  cg.incremental_forward(loss);
	  cg.backward(loss);
  }
  dynet::autobatch_flag = autobatch_cache;
}

BOOST_AUTO_TEST_CASE( loss_gradient ) {
  auto autobatch_cache = dynet::autobatch_flag;
  for(size_t i = 0; i < 2; ++i) {
    dynet::autobatch_flag = i;
	  ComputationGraph cg;
	  ParameterCollection model;
	  Parameter param = model.add_parameters({ 1 });

	  Expression loss = zeroes(cg, { 1 });
	  parameter(cg, param);

	  cg.incremental_forward(loss);
	  cg.backward(loss);
    BOOST_CHECK_EQUAL(as_scalar(loss.gradient()), 1.f);

  }
  dynet::autobatch_flag = autobatch_cache;
}

BOOST_AUTO_TEST_SUITE_END()
