#define BOOST_TEST_MODULE TEST_PARAMS

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/model.h>
#include <dynet/param-init.h>
#include <dynet/lstm.h>
#include <boost/test/unit_test.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include "test.h"

using namespace dynet;
using namespace dynet::expr;
using namespace std;

struct ParamsTest {
    ParamsTest() {
        // initialize if necessary
        if (default_device == nullptr) {
            for (auto x : {"ParamsTest", "--dynet-mem", "512"}) {
                av.push_back(strdup(x));
            }
            char **argv = &av[0];
            int argc = av.size();
            dynet::initialize(argc, argv);
        }
        gain = 2.0;
        epsilon = 1e-6; 
        d = dynet::Dim({10, 10});
    }
    ~ParamsTest() {
        for (auto x : av) free(x);
    }


    float gain, epsilon;
    dynet::Dim d;
    std::vector<char*> av;
};

class testModel {
 public:
  testModel(dynet::ParameterCollection &model) {
    lookup_param = model.add_lookup_parameters(1000, {128});
    affine_params = model.add_subcollection("affine");
    W_x = affine_params.add_parameters({40, 30});
    b_x = affine_params.add_parameters({40});
  }
  std::string get_affine_model_name() { return affine_params.get_namespace(); }
  dynet::ParameterCollection get_affine_model() const { return affine_params; }
 private:
  dynet::LookupParameter lookup_param;
  dynet::Parameter W_x, b_x;
  dynet::ParameterCollection affine_params;
  dynet::LSTMBuilder lstm;
}; // class testModel

// define the test suite
BOOST_FIXTURE_TEST_SUITE(params_test, ParamsTest);

BOOST_AUTO_TEST_CASE( init_saxe ) {
    dynet::ParameterCollection mod;
    // Random orthogonal matrix scaled by gain
    dynet::Parameter saxe_p = mod.add_parameters({10, 10}, ParameterInitSaxe(gain));
    // gain^2 * identity matrix
    dynet::Parameter identity_p = mod.add_parameters({10, 10}, ParameterInitIdentity());
    // Initialize graph
    dynet::ComputationGraph cg;
    dynet::Expression saxe = dynet::parameter(cg, saxe_p);
    dynet::Expression identity = dynet::parameter(cg, identity_p);
    // check that the matrix is indeed orthogonal
    dynet::Expression diff_expr_left = dynet::squared_norm(dynet::transpose(saxe) * saxe - (gain * gain) * identity);
    dynet::Expression diff_expr_right = dynet::squared_norm(saxe * dynet::transpose(saxe) - (gain * gain) * identity);
    float diff = dynet::as_scalar(cg.forward((diff_expr_left + diff_expr_right) / 2.0));
    // Leave a margin of error of epsilon=10^-6
    BOOST_CHECK_LT(diff, epsilon);
}

BOOST_AUTO_TEST_CASE ( test_parameter_collection ) {
  dynet::ParameterCollection model;
  dynet::Parameter a = model.add_parameters({10});
  dynet::Parameter b1 = model.add_parameters({1,2}, "b");
  dynet::Parameter b2 = model.add_parameters({1,2}, "b");
  dynet::ParameterCollection submodel = model.add_subcollection("foo");
  dynet::Parameter c = submodel.add_parameters({10});
  dynet::Parameter d = submodel.add_parameters({1, 2}, "d");
  dynet::Parameter b3 = submodel.add_parameters({1, 2}, "b");
  DYNET_CHECK_EQUAL(model.get_namespace(), "/");
  DYNET_CHECK_EQUAL(a.get_fullname(), "/__0");
  DYNET_CHECK_EQUAL(b1.get_fullname(), "/b__0");
  DYNET_CHECK_EQUAL(b2.get_fullname(), "/b__1");
  DYNET_CHECK_EQUAL(submodel.get_namespace(), "/foo__0/");
  DYNET_CHECK_EQUAL(c.get_fullname(), "/foo__0/__0");
  DYNET_CHECK_EQUAL(d.get_fullname(), "/foo__0/d__0");
  DYNET_CHECK_EQUAL(b3.get_fullname(), "/foo__0/b__0");
}

BOOST_AUTO_TEST_CASE ( test_parameter_class ) {
  auto save_parameters_lambda = [] (const std::string & fname, dynet::ParameterCollection & m) -> size_t {
    auto params = m.get_parameters();
    auto lookup_params = m.get_lookup_parameters();
    for (auto & param: params) {
      std::cout << param->name << "saved in file " << fname << std::endl;
    }
    for (auto & lookup_param: lookup_params) {
      std::cout << lookup_param->name << "saved in file " << fname << std::endl;
    }
    return params.size() + lookup_params.size();
  };
  auto save_parameters_lambda2 = [] (const std::string & fname, dynet::Parameter & p) -> std::string {
    std::cout << p.get_storage().name << "saved in file " << fname << std::endl;
    return p.get_storage().name;
  };
  ParameterCollection collec;
  testModel spec(collec);
  std::string affine_id_for_posterity = spec.get_affine_model_name();
  DYNET_CHECK_EQUAL(affine_id_for_posterity, "/affine__0/");
  DYNET_CHECK_EQUAL(save_parameters_lambda("model_file.txt", collec), 3);
  auto affine_model = spec.get_affine_model();
  DYNET_CHECK_EQUAL(save_parameters_lambda("affine_file.txt", affine_model), 2);
  auto submodel = collec.add_subcollection("affine");
  auto p = submodel.add_parameters({10});
  DYNET_CHECK_EQUAL(save_parameters_lambda2("tuning_parameter_file.txt", p), "/affine__1/__0");
}

BOOST_AUTO_TEST_SUITE_END()
