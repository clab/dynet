#define BOOST_TEST_MODULE TEST_MEM

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/training.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

using namespace dynet;
using namespace std;


struct MemTest {
  MemTest() {
    // initialize if necessary
    for (auto x : {"MemTest", "--dynet-mem", "3"}) {
      av.push_back(strdup(x));
    }
    char **argv = &av[0];
    int argc = av.size();
    dynet::initialize(argc, argv);
  }
  ~MemTest() {
    for (auto x : av) free(x);
  }

  std::vector<char*> av;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(mem_test, MemTest);

BOOST_AUTO_TEST_CASE( expand_test ) {
  if(!autobatch_flag) {
    dynet::ParameterCollection mod;
    dynet::Parameter param = mod.add_parameters({1024,1024});
    SimpleSGDTrainer trainer(mod);
    dynet::ComputationGraph cg;
    Expression x = parameter(cg, param);
    Expression z = sum_rows(sum_cols(x));
    cg.forward(z);
    cg.backward(z);
    trainer.update(0.1);
  }
}

BOOST_AUTO_TEST_SUITE_END();
