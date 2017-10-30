#define BOOST_TEST_MODULE TEST_MEM

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/training.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include "test.h"
#include <stdexcept>

using namespace dynet;
using namespace std;


struct MemTest {
  MemTest() {
    // initialize if necessary
    for (auto x : {"MemTest", "--dynet-mem", "4"}) {
      av.push_back(strdup(x));
    }
    ADD_EXTRA_ARGUMENTS(av)
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
    Expression z = sum_elems(x);
    cg.forward(z);
    cg.backward(z);
    trainer.update();
  }
}

BOOST_AUTO_TEST_SUITE_END();
