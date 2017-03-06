#define BOOST_TEST_MODULE TEST_PARAMS

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/model.h>
#include <boost/test/unit_test.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <iostream>
#include <fstream>
#include <initializer_list>

#include <stdexcept>

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
        pic0 = new dynet::ParameterInitConst(0);
        pic1 = new dynet::ParameterInitConst(1);
        pic3 = new dynet::ParameterInitConst(3);
        inits1.push_back(pic0);
        inits1.push_back(pic1);
        inits1.push_back(pic3);
        inits2.push_back(pic0);
        inits2.push_back(pic1);

        dims1 = {10, 10, 5};
        dims2 = {10, 10};

    }
    ~ParamsTest() {
        delete pic0;
        delete pic1;
        delete pic3;
        for (auto x : av) free(x);
    }

    vector<unsigned> dims1, dims2;
    vector<dynet::ParameterInit*> inits1;
    vector<dynet::ParameterInit*> inits2;
    dynet::ParameterInit* pic0;
    dynet::ParameterInit* pic1;
    dynet::ParameterInit* pic3;
    float gain, epsilon;
    dynet::Dim d;
    std::vector<char*> av;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(params_test, ParamsTest);

BOOST_AUTO_TEST_CASE( init_saxe ) {
    dynet::Model mod;
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

BOOST_AUTO_TEST_CASE( init_joint_matrix ) {
    dynet::Model mod;
    // Create parameter
    std::cerr << "init_joint_matrix" << std::endl;
    dynet::Parameter mat_p = mod.add_parameters({25, 10}, ParameterInitJoint(dims1, inits1));
    std::cerr << "init done" << std::endl;
    // Initialize graph
    dynet::ComputationGraph cg;
    // Load param
    dynet::Expression mat = dynet::parameter(cg, mat_p);
    // Compute the norm
    dynet::Expression norm0_expr = dynet::squared_norm(select_rows(mat, std::vector<unsigned> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
    float norm0 = as_scalar(norm0_expr.value());
    // Check equality
    BOOST_CHECK_CLOSE(norm0, 0, 0.0001);
    // Compute the norm
    dynet::Expression norm1_expr = dynet::squared_norm(select_rows(mat, std::vector<unsigned> {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
    float norm1 = as_scalar(norm1_expr.value());
    // Check equality
    BOOST_CHECK_CLOSE(norm1, 100, 0.0001);
    // Compute the norm
    dynet::Expression norm3_expr = dynet::squared_norm(select_rows(mat, std::vector<unsigned> {20, 21, 22, 23, 24}));
    float norm3 = as_scalar(norm3_expr.value());
    // Check equality
    BOOST_CHECK_CLOSE(norm3, 450, 0.0001);
}

BOOST_AUTO_TEST_CASE( init_joint_vector ) {
    dynet::Model mod;
    // Create parameter
    dynet::Parameter vec_p = mod.add_parameters({25}, ParameterInitJoint(dims1, inits1));
    // Initialize graph
    dynet::ComputationGraph cg;
    // Load param
    dynet::Expression vec = dynet::parameter(cg, vec_p);
    // Compute the norm
    dynet::Expression norm0_expr = dynet::squared_norm(pickrange(vec, 0, 9));
    float norm0 = as_scalar(norm0_expr.value());
    // Check equality
    BOOST_CHECK_CLOSE(norm0, 0, 0.0001);
    // Compute the norm
    dynet::Expression norm1_expr = dynet::squared_norm(pickrange(vec, 10, 20));
    float norm1 = as_scalar(norm1_expr.value());
    // Check equality
    BOOST_CHECK_CLOSE(norm1, 10, 0.0001);
    // Compute the norm
    dynet::Expression norm3_expr = dynet::squared_norm(pickrange(vec, 20, 25));
    float norm3 = as_scalar(norm3_expr.value());
    // Check equality
    BOOST_CHECK_CLOSE(norm3, 45, 0.0001);
}

BOOST_AUTO_TEST_CASE( init_joint_except_vec ) {
    dynet::Model mod;
    // Create parameter
    BOOST_CHECK_THROW(mod.add_parameters({15}, ParameterInitJoint(dims2, inits2)), std::invalid_argument)
}

BOOST_AUTO_TEST_CASE( init_joint_except_mat ) {
    dynet::Model mod;
    // Create parameter
    BOOST_CHECK_THROW(mod.add_parameters({15, 2}, ParameterInitJoint(dims2, inits2)), std::invalid_argument)
}

BOOST_AUTO_TEST_CASE( init_joint_except_tensor ) {
    dynet::Model mod;
    // Create parameter
    BOOST_CHECK_THROW(mod.add_parameters({15, 2, 3}, ParameterInitJoint(dims2, inits2)), std::invalid_argument)
}

BOOST_AUTO_TEST_SUITE_END()
