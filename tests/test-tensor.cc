#define BOOST_TEST_MODULE TEST_NODES

#include <dynet/functors.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>
#include <dynet/index-tensor.h>
#include <boost/test/unit_test.hpp>
#include "test.h"
#include <stdexcept>

using namespace dynet;
using namespace std;


struct TensorTest {
  TensorTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"TensorTest", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      ADD_EXTRA_ARGUMENTS(av)
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }

    batch_vals = {1.f, 2.f, 1.f, 4.f, 5.f, 6.f};
  }
  ~TensorTest() {
    for (auto x : av) free(x);
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
  std::vector<float> batch_vals;
  dynet::ParameterCollection mod;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(tensor_test, TensorTest);

// test whether argmax is working properly
BOOST_AUTO_TEST_CASE( argmax ) {
  dynet::ComputationGraph cg;
  Expression x1 = input(cg, Dim({ 3 }, 2), batch_vals);
  IndexTensor idx_tens = TensorTools::argmax(x1.value());
  vector<Eigen::DenseIndex> idx_act = as_vector(idx_tens);
  vector<Eigen::DenseIndex> idx_exp = {1, 2};
  BOOST_CHECK_EQUAL_COLLECTIONS(idx_exp.begin(), idx_exp.end(),
                                idx_act.begin(), idx_act.end());
}

// test whether topk is working properly
BOOST_AUTO_TEST_CASE(topk) {
  dynet::ComputationGraph cg;
  Expression x1 = input(cg, Dim({3}, 2), batch_vals);
  std::pair<Tensor, IndexTensor> rets = TensorTools::topk(x1.value());
  // idx
  vector<Eigen::DenseIndex> idx_act = as_vector(rets.second);
  vector<Eigen::DenseIndex> idx_exp = {1, 2};
  BOOST_CHECK_EQUAL_COLLECTIONS(idx_exp.begin(), idx_exp.end(),
    idx_act.begin(), idx_act.end());
  // val
  vector<float> val_act = as_vector(rets.first);
  vector<float> val_exp = {2.f, 6.f};
  BOOST_CHECK_EQUAL_COLLECTIONS(val_act.begin(), val_act.end(),
    val_exp.begin(), val_exp.end());
}

// for now, just make sure that things don't die
BOOST_AUTO_TEST_CASE( categorical_sample_log_prob ) {
  dynet::ComputationGraph cg;
  Expression x1 = input(cg, Dim({ 3 }, 2), batch_vals);
  IndexTensor idx_tens = TensorTools::categorical_sample_log_prob(x1.value());
  vector<Eigen::DenseIndex> idx_act = as_vector(idx_tens);
  vector<Eigen::DenseIndex> idx_exp = {0, 2};
}

BOOST_AUTO_TEST_SUITE_END()
