#define BOOST_TEST_MODULE TEST_PARAMS

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

#include <boost/test/unit_test.hpp>

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/model.h>
#include <dynet/param-init.h>
#include <dynet/lstm.h>
#include <dynet/gru.h>
#include <dynet/treelstm.h>
#include <dynet/io.h>

#include "test.h"

using namespace dynet;
using namespace std;

struct ParamsTest {
    ParamsTest() {
        // initialize if necessary
        if (default_device == nullptr) {
            for (auto x : {"ParamsTest", "--dynet-mem", "512"}) {
                av.push_back(strdup(x));
            }
            ADD_EXTRA_ARGUMENTS(av)
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

class testParameterCollection {
 public:
  testParameterCollection(dynet::ParameterCollection &model) {
    lookup_param = model.add_lookup_parameters(1000, {128});
    affine_params = model.add_subcollection("affine");
    W_x = affine_params.add_parameters({40, 30});
    b_x = affine_params.add_parameters({40});
  }
  std::string get_affine_model_name() { return affine_params.get_fullname(); }
  dynet::ParameterCollection get_affine_model() const { return affine_params; }
 private:
  dynet::LookupParameter lookup_param;
  dynet::Parameter W_x, b_x;
  dynet::ParameterCollection affine_params;
}; // class testParameterCollection

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
  DYNET_CHECK_EQUAL(model.get_fullname(), "/");
  DYNET_CHECK_EQUAL(a.get_fullname(), "/_0");
  DYNET_CHECK_EQUAL(b1.get_fullname(), "/b");
  DYNET_CHECK_EQUAL(b2.get_fullname(), "/b_1");
  DYNET_CHECK_EQUAL(submodel.get_fullname(), "/foo/");
  DYNET_CHECK_EQUAL(c.get_fullname(), "/foo/_0");
  DYNET_CHECK_EQUAL(d.get_fullname(), "/foo/d");
  DYNET_CHECK_EQUAL(b3.get_fullname(), "/foo/b");
}

BOOST_AUTO_TEST_CASE ( test_parameter_class ) {
  auto save_parameters_lambda = [] (const std::string & fname, dynet::ParameterCollection & m) -> size_t {
    auto params = m.get_parameter_storages();
    auto lookup_params = m.get_lookup_parameter_storages();
    for (auto & param: params) {
      std::cout << param->name << " saved in file " << fname << std::endl;
    }
    for (auto & lookup_param: lookup_params) {
      std::cout << lookup_param->name << " saved in file " << fname << std::endl;
    }
    return params.size() + lookup_params.size();
  };
  auto save_parameters_lambda2 = [] (const std::string & fname, dynet::Parameter & p) -> std::string {
    std::cout << p.get_storage().name << " saved in file " << fname << std::endl;
    return p.get_storage().name;
  };
  auto save_parameters_lambda3 = [] (const std::string & fname,
                                     std::shared_ptr<dynet::ParameterStorage>p) ->std::string {
    std::cout << p->name << " saved in file " << fname << std::endl;
    return p->name;
  };
  ParameterCollection collec;
  testParameterCollection spec(collec);
  std::string affine_id_for_posterity = spec.get_affine_model_name();
  DYNET_CHECK_EQUAL(affine_id_for_posterity, "/affine/");
  DYNET_CHECK_EQUAL(save_parameters_lambda("model_file.txt", collec), 3);
  auto affine_model = spec.get_affine_model();
  DYNET_CHECK_EQUAL(save_parameters_lambda("affine_file.txt", affine_model), 2);
  auto submodel = collec.add_subcollection("affine");
  auto p = submodel.add_parameters({10});
  std::cout << p.get_fullname() << std::endl;
  DYNET_CHECK_EQUAL(save_parameters_lambda2("tuning_parameter_file.txt", p), "/affine_1/_0");
  DYNET_CHECK_EQUAL(save_parameters_lambda3("tuning_parameter_file.txt",
                                            affine_model.get_parameter_storage("/affine/_1")), "/affine/_1");
}

BOOST_AUTO_TEST_CASE ( test_parametercollection_with_builder ) {
  dynet::ParameterCollection collec;
  auto gru_builder = dynet::GRUBuilder(3, 10, 2, collec);
  DYNET_CHECK_EQUAL(gru_builder.get_parameter_collection().size(), 9 * 3);
  dynet::ParameterCollection collec2;
  auto bi_treelstm_builder = BidirectionalTreeLSTMBuilder(3, 10, 2, collec2);
  DYNET_CHECK_EQUAL(bi_treelstm_builder.get_parameter_collection().size(), 3 * 3 * 2);
}

BOOST_AUTO_TEST_CASE( scale ) {
    dynet::ParameterCollection mod;
    // Create parameter
    dynet::Parameter w_p = mod.add_parameters({1}, ParameterInitConst(1));
    // Initial value
    float init_value= as_scalar(*(w_p.values()));
    // Rescale
    w_p.scale(0.3);
    // Value after rescaling
    float end_value=as_scalar(*(w_p.values()));
    // Check with a margin of error
    BOOST_CHECK_CLOSE(init_value * 0.3, end_value, 0.001);
}

BOOST_AUTO_TEST_CASE( scale_grad ) {
    dynet::ParameterCollection mod;
    // Create parameter
    dynet::Parameter w_p = mod.add_parameters({1}, ParameterInitConst(1));
    // Run forward/backward
    dynet::ComputationGraph cg;
    dynet::Expression x = dynet::input(cg,1.f);
    dynet::Expression w = dynet::parameter(cg, w_p);
    dynet::Expression y = x * w;
    cg.forward(y);
    cg.backward(y);
    // Rescale gradient
    w_p.scale_gradient(0.5);
    // Value after rescaling
    float rescaled_grad=as_scalar(w_p.get_storage().g);
    // Check with a margin of error
    BOOST_CHECK_CLOSE(0.5, rescaled_grad, 0.001);
}

BOOST_AUTO_TEST_CASE( set_value ) {
    dynet::ParameterCollection mod;
    // Create parameter
    dynet::Parameter w_p = mod.add_parameters({3}, ParameterInitConst(1));
    // Initial parameter = {1., 1., 1.}
    std::vector<float> value_to_set = {1.f, 2.f, 3.f};
    // Set value
    w_p.set_value(value_to_set);
    // New value
    std::vector<float> parameter_after = as_vector(w_p.get_storage().values);
    for(unsigned i=0; i<3; i++){
      BOOST_CHECK_EQUAL(value_to_set[i], parameter_after[i]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
