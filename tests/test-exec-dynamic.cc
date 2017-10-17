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
#include <thread>

using namespace dynet;
using namespace std;


struct ExecTest {
  ExecTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"ExecTest", "--dynet-seed", "10", "--dynet-mem", "1",
                     "--dynet-dynamic-mem", "1"}) {
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

BOOST_AUTO_TEST_CASE( lstm ) {
  vector<float> results;
  dynet::ParameterCollection mod;
  dynet::VanillaLSTMBuilder lstm(2, 3, 10, mod);
  dynet::LookupParameter lp = mod.add_lookup_parameters(10, {3});
  dynet::autobatch_flag = 0;
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

  for (size_t i = 1; i < results.size(); ++i)
    BOOST_CHECK_CLOSE(results[0], results[i], 0.0001);
}

BOOST_AUTO_TEST_CASE( param_after_node ) {
  auto autobatch_cache = dynet::autobatch_flag;
  dynet::autobatch_flag = 0;
  ComputationGraph cg;
  ParameterCollection model;
  Parameter param = model.add_parameters({ 1 });
  
  Expression loss = zeroes(cg, { 1 });
  parameter(cg, param);
  
  cg.incremental_forward(loss);
  cg.backward(loss);
  dynet::autobatch_flag = autobatch_cache;
}

BOOST_AUTO_TEST_CASE( param_after_node_2 ) {
  auto autobatch_cache = dynet::autobatch_flag;
  dynet::autobatch_flag = 0;
  ComputationGraph cg;
  ParameterCollection model;
  LookupParameter param = model.add_lookup_parameters(10, { 1 });

  lookup(cg, param, 1);
  Expression loss = zeroes(cg, { 1 });

  cg.incremental_forward(loss);
  cg.backward(loss);
  dynet::autobatch_flag = autobatch_cache;
}

BOOST_AUTO_TEST_CASE( multi_lstm ) {
  vector<vector<float>> results(4);
  dynet::ParameterCollection mod;
  dynet::VanillaLSTMBuilder lstm_proto(2, 3, 10, mod);
  dynet::LookupParameter lp_proto = mod.add_lookup_parameters(10, {3});
  dynet::autobatch_flag = 0;
  
  vector<thread> threads(4);
  for (size_t t = 0; t < 4; ++t) {
    threads[t] = thread([&, t]() {
      dynet::ComputationGraph cg;
      dynet::VanillaLSTMBuilder lstm(lstm_proto);
      dynet::LookupParameter lp(lp_proto);
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
      results[t].push_back(as_scalar(z.value()));
    });
  }

  for (size_t t = 0; t < 4; ++t) { threads[t].join(); }
  for (size_t t = 0; t < 4; ++t) {
    for(size_t i = 1; i < results[t].size(); ++i) {
      BOOST_CHECK_CLOSE(results[t][0], results[t][i], 0.0001);
    }
  }
}


BOOST_AUTO_TEST_SUITE_END()
