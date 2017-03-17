#define BOOST_TEST_MODULE TEST_PARAMS

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/model.h>
#include <dynet/param-init.h>
#include <boost/test/unit_test.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <iostream>
#include <fstream>

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
    }
    ~ParamsTest() {
        for (auto x : av) free(x);
    }


    float gain, epsilon;
    dynet::Dim d;
    std::vector<char*> av;
};

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

BOOST_AUTO_TEST_SUITE_END()
