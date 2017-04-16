#define BOOST_TEST_MODULE TEST_NODES

#include <dynet/functors.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

using namespace dynet;
using namespace dynet::expr;
using namespace std;


struct TensorTest {
  TensorTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"TensorTest", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
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
  dynet::Model mod;
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
  BOOST_TEST(idx_exp == idx_act, boost::test_tools::per_element() );
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
