#define BOOST_TEST_MODULE TEST_RNN

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/cfsm-builder.h>
#include <dynet/hsm-builder.h>
#include <dynet/dict.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include "test.h"
#include <stdexcept>
#include <fstream>
#include <sstream>


using namespace dynet;
using namespace std;


struct SoftmaxTest {
  SoftmaxTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"SoftmaxTest", "--dynet-seed", "10", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      ADD_EXTRA_ARGUMENTS(av)
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }
    vec_values = {1.f, 2.f, 3.f};
    batch_values = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    ofstream ofs("cluster_file.txt");
    for (unsigned i = 0; i < 5; ++i) {
      for (unsigned j = 0; j < 2; ++j) {
        stringstream ss;
        ss << (i * 2 + j);
        dic.convert(ss.str());
        ofs << i << " " << ss.str() << endl;
      }
    }
    ofs.close();
  }
  ~SoftmaxTest() {
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

  std::vector<float> vec_values, batch_values;
  std::vector<char*> av;
  Dict dic;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(softmax_test, SoftmaxTest);

BOOST_AUTO_TEST_CASE( standard_softmax_grad ) {
  dynet::ParameterCollection mod;
  StandardSoftmaxBuilder softmax(3, 10, mod, true);
  dynet::ComputationGraph cg;
  softmax.new_graph(cg);
  auto h = input(cg, {3}, vec_values);
  auto nll = softmax.neg_log_softmax(h, 5);
  BOOST_CHECK(check_grad(mod, nll, 0));
}

BOOST_AUTO_TEST_CASE( standard_softmax_batch_grad ) {
  dynet::ParameterCollection mod;
  StandardSoftmaxBuilder softmax(3, 10, mod, true);
  dynet::ComputationGraph cg;
  softmax.new_graph(cg);
  Expression h = input(cg, Dim({3}, 2), batch_values);
  vector<unsigned> batch_elems = {2, 5};
  Expression nll = sum_batches(softmax.neg_log_softmax(h, batch_elems));
  BOOST_CHECK(check_grad(mod, nll, 0));
}

BOOST_AUTO_TEST_CASE( cf_softmax_grad ) {
  dynet::ParameterCollection mod;
  ClassFactoredSoftmaxBuilder softmax(3, "cluster_file.txt", dic, mod, true);
  dynet::ComputationGraph cg;
  softmax.new_graph(cg);
  auto h = input(cg, {3}, vec_values);
  auto nll = softmax.neg_log_softmax(h, 5);
  BOOST_CHECK(check_grad(mod, nll, 0));
}

BOOST_AUTO_TEST_CASE( cf_softmax_batch_grad ) {
  dynet::ParameterCollection mod;
  ClassFactoredSoftmaxBuilder softmax(3, "cluster_file.txt", dic, mod, true);
  dynet::ComputationGraph cg;
  softmax.new_graph(cg);
  Expression h = input(cg, Dim({3}, 2), batch_values);
  vector<unsigned> batch_elems = {2, 5};
  Expression nll = sum_batches(softmax.neg_log_softmax(h, batch_elems));
  BOOST_CHECK(check_grad(mod, nll, 0));
}

BOOST_AUTO_TEST_SUITE_END()
