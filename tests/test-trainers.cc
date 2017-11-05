#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/training.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

using namespace dynet;
using namespace dynet::expr;
using namespace std;


struct TrainerTest {
  TrainerTest() {
    // initialize if necessary
    if(default_device == nullptr) {
      for (auto x : {"TrainerTest", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }
    ones_vals = {1.f,1.f,1.f};
    param_vals = {1.1f,-2.2f,3.3f};
  }
  ~TrainerTest() {
    for (auto x : av) free(x);
  }

  template <class T>
  std::string print_vec(const std::vector<T> vec) {
    ostringstream oss;
    if(vec.size()) oss << vec[0];
    for(size_t i = 1; i < vec.size(); i++)
      oss << ' ' << vec[i];
    return oss.str();
  }

  std::vector<float> ones_vals, param_vals;
  std::vector<char*> av;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(trainer_test, TrainerTest);

BOOST_AUTO_TEST_CASE( simple_sgd_direction ) {
  dynet::Model mod;
  dynet::Parameter param = mod.add_parameters({3});
  TensorTools::SetElements(param.get()->values,param_vals);
  SimpleSGDTrainer trainer(mod);
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
