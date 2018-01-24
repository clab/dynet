#define BOOST_TEST_MODULE TEST_TRAINERS

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/training.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include "test.h"
#include <stdexcept>

using namespace dynet;
using namespace std;


struct TrainerTest {
  TrainerTest() {
    // initialize if necessary
    if(default_device == nullptr) {
      for (auto x : {"TrainerTest", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      ADD_EXTRA_ARGUMENTS(av)
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }
    ones_vals = {1.f,1.f,1.f};
    param_vals = {1.1f,-2.2f,3.3f};
    param2_vals = {1.1f,-2.2f,3.3f};
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

  std::vector<float> ones_vals, param_vals, param2_vals;
  std::vector<char*> av;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(trainer_test, TrainerTest);

// Test direction (does updating reduce the objective value)

#define DYNET_TRAINER_DIRECTION_TEST_CASE(name, TRAINER_TYPE)       \
BOOST_AUTO_TEST_CASE( name ) {                                      \
  dynet::ParameterCollection mod;                                   \
  dynet::Parameter param = mod.add_parameters({3});                 \
  TensorTools::set_elements(param.get_storage().values, param_vals);\
  TRAINER_TYPE trainer(mod);                                        \
  dynet::ComputationGraph cg;                                       \
  Expression x = parameter(cg, param);                              \
  Expression y = input(cg, {1, 3}, ones_vals);                      \
  Expression z = y * x;                                             \
  float before = as_scalar(cg.forward(z));                          \
  cg.backward(z);                                                   \
  trainer.update();                                                 \
  float after = as_scalar(cg.forward(z));                           \
  BOOST_CHECK_LT(after, before);                                    \
}                                                                   \

DYNET_TRAINER_DIRECTION_TEST_CASE(simple_sgd_direction, dynet::SimpleSGDTrainer)

DYNET_TRAINER_DIRECTION_TEST_CASE(cyclical_sgd_direction, dynet::CyclicalSGDTrainer)

DYNET_TRAINER_DIRECTION_TEST_CASE(momentum_sgd_direction, dynet::MomentumSGDTrainer)

DYNET_TRAINER_DIRECTION_TEST_CASE(adagrad_direction, dynet::AdagradTrainer)

DYNET_TRAINER_DIRECTION_TEST_CASE(adadelta_direction, dynet::AdadeltaTrainer)

DYNET_TRAINER_DIRECTION_TEST_CASE(rmsprop_direction, dynet::RMSPropTrainer)

DYNET_TRAINER_DIRECTION_TEST_CASE(adam_direction, dynet::AdamTrainer)

DYNET_TRAINER_DIRECTION_TEST_CASE(amsgrad_direction, dynet::AmsgradTrainer)


BOOST_AUTO_TEST_CASE( eg_direction ) {
  param_vals = {0.5f,0.1f,0.4f};// EGTrainer requires values belonging to simplex [0,1]
  dynet::ParameterCollection mod;
  dynet::Parameter param = mod.add_parameters({3});
  TensorTools::set_elements(param.get_storage().values,param_vals);
  EGTrainer trainer(mod);
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param);
  Expression y = input(cg, {1,3}, ones_vals);
  Expression z = y*x;
  float before = as_scalar(cg.forward(z));
  cg.backward(z);
  trainer.update();
  float after = as_scalar(cg.forward(z));
  BOOST_CHECK_EQUAL(after, before);
  param_vals = {1.1f,-2.2f,3.3f};// revert back to original values
}

// Test the restart method (TODO: for now only checks for errors)

#define DYNET_TRAINER_RESTART_TEST_CASE(name, TRAINER_TYPE)         \
BOOST_AUTO_TEST_CASE( name ) {                                      \
  dynet::ParameterCollection mod;                                   \
  dynet::Parameter param = mod.add_parameters({3});                 \
  TensorTools::set_elements(param.get_storage().values, param_vals);\
  TRAINER_TYPE trainer(mod);                                        \
  dynet::ComputationGraph cg;                                       \
  Expression x = parameter(cg, param);                              \
  Expression y = input(cg, {1, 3}, ones_vals);                      \
  Expression z = y * x;                                             \
  cg.forward(z);                                                    \
  cg.backward(z);                                                   \
  trainer.update();                                                 \
  trainer.restart(0.1);                                             \
}                                                                   \

DYNET_TRAINER_RESTART_TEST_CASE(simple_sgd_restart, dynet::SimpleSGDTrainer)

DYNET_TRAINER_RESTART_TEST_CASE(cyclical_sgd_restart, dynet::CyclicalSGDTrainer)

DYNET_TRAINER_RESTART_TEST_CASE(momentum_sgd_restart, dynet::MomentumSGDTrainer)

DYNET_TRAINER_RESTART_TEST_CASE(adagrad_restart, dynet::AdagradTrainer)

DYNET_TRAINER_RESTART_TEST_CASE(adadelta_restart, dynet::AdadeltaTrainer)

DYNET_TRAINER_RESTART_TEST_CASE(rmsprop_restart, dynet::RMSPropTrainer)

DYNET_TRAINER_RESTART_TEST_CASE(adam_restart, dynet::AdamTrainer)

DYNET_TRAINER_RESTART_TEST_CASE(amsgrad_restart, dynet::AmsgradTrainer)

DYNET_TRAINER_RESTART_TEST_CASE(eg_restart, dynet::EGTrainer)

BOOST_AUTO_TEST_CASE( momentum_restart_correctness ) {
  dynet::ParameterCollection pc;
  dynet::Parameter param = pc.add_parameters({3});
  MomentumSGDTrainer trainer(pc, 1.0);
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param);
  Expression y = input(cg, {1,3}, ones_vals);
  Expression z = y * x1;
  cg.backward(z);
  trainer.update();
  vector<float> vp_val = as_vector(trainer.vp[0].h);
  // Test that the velocity has been updated
  for(size_t i = 0; i < vp_val.size(); ++i)
    BOOST_CHECK_EQUAL(vp_val[i], -1.0);
  trainer.restart(0.5);
  vp_val = as_vector(trainer.vp[0].h);
  // Test that the velocity has been reset to 0
  for(size_t i = 0; i < vp_val.size(); ++i)
    BOOST_CHECK_EQUAL(vp_val[i], 0);
  // Test that the learning rate has been set to 0.5
  BOOST_CHECK_EQUAL(trainer.learning_rate, 0.5);
}

// Test subset update

BOOST_AUTO_TEST_CASE( simple_sgd_update_subset ) {
  dynet::ParameterCollection mod;
  dynet::Parameter param = mod.add_parameters({3});
  dynet::Parameter param2 = mod.add_parameters({3});
  TensorTools::set_elements(param.get_storage().values,param_vals);
  TensorTools::set_elements(param2.get_storage().values,param2_vals);
  param2.set_updated(false);
  SimpleSGDTrainer trainer(mod);
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param);
  Expression x2 = parameter(cg, param2);
  Expression y = input(cg, {1,3}, ones_vals);
  Expression z = y*(x1+x2);
  cg.backward(z);
  trainer.update();
  vector<float> param_after = as_vector(param.get_storage().values);
  vector<float> param2_after = as_vector(param2.get_storage().values);
  for(size_t i = 0; i < param_after.size(); ++i)
    BOOST_CHECK_NE(param_vals[i], param_after[i]);
  for(size_t i = 0; i < param2_after.size(); ++i)
    BOOST_CHECK_EQUAL(param2_vals[i], param2_after[i]);
}

BOOST_AUTO_TEST_CASE( simple_edge_case_test_dynet_867 ) {
  ParameterCollection model;
  LookupParameter emb_weight = model.add_lookup_parameters(1, {1});
  model.add_parameters({1, 1});
  SimpleSGDTrainer trainer(model);
  ComputationGraph cg;
  dynet::real zero = 0.0;
  Expression loss2 = input(cg, zero);
  for (int i = 0; i < 1000; ++i)
    dynet::lookup(cg, emb_weight, 0u);
  cg.backward(loss2);
  trainer.update();
}

BOOST_AUTO_TEST_SUITE_END()
