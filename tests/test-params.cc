#define BOOST_TEST_MODULE TEST_PARAMS

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

#include <boost/test/unit_test.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

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
}; // class testModel

class testModel2 {
 public:
  testModel2(dynet::ParameterCollection &model) {
    lookup_param = model.add_lookup_parameters(1000, {128});
    affine_params = model.add_subcollection("affine");
    W_x = affine_params.add_parameters({40, 30});
    b_x = affine_params.add_parameters({40});
    lstm = LSTMBuilder(3, 40, 1, model);
  }
  std::string get_affine_model_name() { return affine_params.get_namespace(); }
  dynet::ParameterCollection get_affine_model() const { return affine_params; }
  dynet::ParameterCollection get_lstm_model() { return lstm.get_parameters(); }
 private:
  dynet::LookupParameter lookup_param;
  dynet::Parameter W_x, b_x;
  dynet::ParameterCollection affine_params;
  dynet::LSTMBuilder lstm;
}; // class testModel2

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
                                     dynet::ParameterStorage *p) ->std::string {
    std::cout << p->name << " saved in file " << fname << std::endl;
    return p->name;
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
  std::cout << p.get_fullname() << std::endl;
  DYNET_CHECK_EQUAL(save_parameters_lambda2("tuning_parameter_file.txt", p), "/affine__1/__0");
  DYNET_CHECK_EQUAL(save_parameters_lambda3("tuning_parameter_file.txt",
                                            affine_model.get_parameter_storage("/affine__0/__0")), "/affine__0/__0");
}

BOOST_AUTO_TEST_CASE ( test_parameter_class_with_builder ) {
  auto save_parameters_lambda = [] (const std::string & fname,
                                    ParameterCollection & param_list) -> size_t {
     for (auto & param : param_list.get_parameter_storages()) {
       std::cout << param->name << " saved in file " << fname << std::endl;
     }
     return param_list.size();
  };
  ParameterCollection collec;
  testModel2 spec(collec);
  auto params = spec.get_lstm_model();
  save_parameters_lambda("lstm_file.txt", params);
}

BOOST_AUTO_TEST_CASE ( test_parametercollection_with_builder ) {
  dynet::ParameterCollection collec;
  auto gru_builder = dynet::GRUBuilder(3, 10, 2, collec);
  DYNET_CHECK_EQUAL(gru_builder.get_parameters().size(), 9 * 3);
  dynet::ParameterCollection collec2;
  auto bi_treelstm_builder = BidirectionalTreeLSTMBuilder(3, 10, 2, collec2);
  DYNET_CHECK_EQUAL(bi_treelstm_builder.get_parameters().size(), 11 * 3 * 2);
}

BOOST_AUTO_TEST_CASE ( test_save_load_parameter_collection ) {
  ParameterCollection m;
  Parameter a = m.add_parameters({10}, "a");
  Parameter b = m.add_parameters({3,7});
  LookupParameter c = m.add_lookup_parameters(10, {2});
  std::remove("test.model"); std::remove("test.model.meta");
  dynet::Pack s("test.model");
  s.save(m, "model1");
  s.save(m, m.get_namespace(), true);

  ParameterCollection m2;
  s.load(m2, "model1");
  auto params1 = m2.get_parameter_storages();
  for(auto & x : params1) {
    std::cout << x->name << std::endl;
    std::cout << x->dim << std::endl;
    std::cout << x->values << std::endl;
    std::cout << x->g << std::endl;
  }
  auto params11 = m.get_parameter_storages();
  for(auto & x : params11) {
    std::cout << x->name << std::endl;
    std::cout << x->dim << std::endl;
    std::cout << x->values << std::endl;
    std::cout << x->g << std::endl;
  }
  ParameterCollection m3;
  s.load(m3, "/");
  auto params2 = m3.get_parameter_storages();
  for(auto & x : params2) {
    std::cout << x->name << std::endl;
    std::cout << x->dim << std::endl;
    std::cout << x->values << std::endl;
    std::cout << x->g << std::endl;
  }
  auto lookup_params = m2.get_lookup_parameter_storages();
  for(auto & x : lookup_params) {
    std::cout << x->name << std::endl;
    std::cout << x->dim << std::endl;
    std::cout << x->all_dim << std::endl;
    std::cout << x->all_values << std::endl;
    std::cout << x->all_grads << std::endl;
    std::cout << x->values[0] << std::endl;
    std::cout << x->grads[0] << std::endl;
  }
  auto lookup_paramss = m.get_lookup_parameter_storages();
  for(auto & x : lookup_paramss) {
    std::cout << x->name << std::endl;
    std::cout << x->dim << std::endl;
    std::cout << x->all_dim << std::endl;
    std::cout << x->all_values << std::endl;
    std::cout << x->all_grads << std::endl;
    std::cout << x->values[0] << std::endl;
    std::cout << x->grads[0] << std::endl;
  }
}

BOOST_AUTO_TEST_CASE ( test_save_load_parameter_collections ) {
  {
    ParameterCollection collec;
    testModel2 spec(collec);
    std::remove("a.model"); std::remove("a.model.meta");
    Pack s1("a.model");
    s1.save(collec, "all");
    ParameterCollection collec2;
    s1.load(collec2);
    DYNET_CHECK_EQUAL(collec2.size(), collec.size());
  
    std::remove("b.model"); std::remove("b.model.meta");
    Pack s2("b.model");
    s2.save(collec, "all");
    s2.save(spec.get_lstm_model(), "lstm", true);
    s2.save(spec.get_affine_model(), "affine", true);
    ParameterCollection collec3, lstm2, affine2;
    s2.load(affine2, "affine");
    s2.load(collec3, "all");
    s2.load(lstm2, "lstm");
    DYNET_CHECK_EQUAL(affine2.size(), spec.get_affine_model().size());
    DYNET_CHECK_EQUAL(collec3.size(), collec.size());
    DYNET_CHECK_EQUAL(lstm2.size(), spec.get_lstm_model().size());

    std::remove("c.model"); std::remove("c.model.meta");
    s2.reinit("c.model");
    s2.save(lstm2, "lstm");
    s2.save(collec3, "all", true);
    s2.save(affine2, "affine", true);
  }

  {
    ParameterCollection cc;
    auto cc2 = cc.add_subcollection("xx");
    cc2.add_parameters({10});
    std::remove("d.model"); std::remove("d.model.meta");
    Pack s3("d.model");
    s3.save(cc, "key");

    ParameterCollection ccc;
    s3.load(ccc, "key");
    DYNET_CHECK_EQUAL(ccc.size(), cc.size());

    std::remove("e.model"); std::remove("e.model.meta");
    s3.reinit("e.model");
    s3.save(ccc);
  }
}

BOOST_AUTO_TEST_SUITE_END()
