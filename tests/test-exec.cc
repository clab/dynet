#define BOOST_TEST_MODULE TEST_RNN

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/exec.h>
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


struct ExecTest {
  ExecTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"ExecTest", "--dynet-seed", "10", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
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


BOOST_AUTO_TEST_SUITE_END()
