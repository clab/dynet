#define BOOST_TEST_MODULE TEST_TRAINERS

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/training.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

using namespace dynet;
using namespace dynet::expr;
using namespace std;


struct MemTest {
  MemTest() {
    // initialize if necessary
    for (auto x : {"MemTest", "--dynet-mem", "1"}) {
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
  dynet::Model mod;
  dynet::Parameter param = mod.add_parameters({1024,1024});
  SimpleSGDTrainer trainer(mod);
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param);
  Expression z = sum_rows(sum_cols(x));
  float before = as_scalar(cg.forward(z));
  cg.backward(z);
  trainer.update(0.1);
  float after = as_scalar(cg.forward(z));
  BOOST_CHECK_LT(after, before);
}

BOOST_AUTO_TEST_CASE( simple_sgd_update_subset ) {
  dynet::Model mod;
  dynet::Parameter param = mod.add_parameters({3});
  dynet::Parameter param2 = mod.add_parameters({3});
  TensorTools::SetElements(param.get()->values,param_vals);
  TensorTools::SetElements(param2.get()->values,param2_vals);
  vector<unsigned> uparam, ulookup;
  uparam.push_back(param.index);
  SimpleSGDTrainer trainer(mod);
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param);
  Expression x2 = parameter(cg, param2);
  Expression y = input(cg, {1,3}, ones_vals);
  Expression z = y*(x1+x2);
  cg.backward(z);
  trainer.update(uparam, ulookup, 0.1);
  vector<float> param_after = as_vector(param.get()->values);
  vector<float> param2_after = as_vector(param2.get()->values);
  for(size_t i = 0; i < param_after.size(); ++i)
    BOOST_CHECK_NE(param_vals[i], param_after[i]);
  for(size_t i = 0; i < param2_after.size(); ++i)
    BOOST_CHECK_EQUAL(param2_vals[i], param2_after[i]);
}

BOOST_AUTO_TEST_CASE( momentum_sgd_direction ) {
  dynet::Model mod;
  dynet::Parameter param = mod.add_parameters({3});
  TensorTools::SetElements(param.get()->values,param_vals);
  MomentumSGDTrainer trainer(mod);
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param);
  Expression y = input(cg, {1,3}, ones_vals);
  Expression z = y*x;
  float before = as_scalar(cg.forward(z));
  cg.backward(z);
  trainer.update(0.1);
  float after = as_scalar(cg.forward(z));
  BOOST_CHECK_LT(after, before);
}

BOOST_AUTO_TEST_CASE( adagrad_direction ) {
  dynet::Model mod;
  dynet::Parameter param = mod.add_parameters({3});
  TensorTools::SetElements(param.get()->values,param_vals);
  AdagradTrainer trainer(mod);
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param);
  Expression y = input(cg, {1,3}, ones_vals);
  Expression z = y*x;
  float before = as_scalar(cg.forward(z));
  cg.backward(z);
  trainer.update(0.1);
  float after = as_scalar(cg.forward(z));
  BOOST_CHECK_LT(after, before);
}

BOOST_AUTO_TEST_CASE( adadelta_direction ) {
  dynet::Model mod;
  dynet::Parameter param = mod.add_parameters({3});
  TensorTools::SetElements(param.get()->values,param_vals);
  AdadeltaTrainer trainer(mod);
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param);
  Expression y = input(cg, {1,3}, ones_vals);
  Expression z = y*x;
  float before = as_scalar(cg.forward(z));
  cg.backward(z);
  trainer.update(0.1);
  float after = as_scalar(cg.forward(z));
  BOOST_CHECK_LT(after, before);
}

BOOST_AUTO_TEST_CASE( adam_direction ) {
  dynet::Model mod;
  dynet::Parameter param = mod.add_parameters({3});
  TensorTools::SetElements(param.get()->values,param_vals);
  AdamTrainer trainer(mod);
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param);
  Expression y = input(cg, {1,3}, ones_vals);
  Expression z = y*x;
  float before = as_scalar(cg.forward(z));
  cg.backward(z);
  trainer.update(0.1);
  float after = as_scalar(cg.forward(z));
  BOOST_CHECK_LT(after, before);
}

BOOST_AUTO_TEST_SUITE_END()
