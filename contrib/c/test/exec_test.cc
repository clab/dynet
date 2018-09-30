#define BOOST_TEST_MODULE C_EXEC_TEST

#include <dynet_c/init.h>
#include <dynet_c/dim.h>
#include <dynet_c/expr.h>
#include <dynet_c/graph.h>
#include <dynet_c/model.h>
#include <dynet_c/rnn-builder.h>
#include <dynet_c/tensor.h>
#include <vector>
#include <boost/test/unit_test.hpp>
#include "test_utils.h"

struct CExecTest {
  CExecTest() {
    const char *argv[] = {
        "ExecTest",
        "--dynet-seed", "10",
        "--dynet-mem", "10",
        "--dynet-autobatch", "1"
    };
    ::dynetDynetParams_t *params;
    ::dynetExtractDynetParams(7, const_cast<char**>(argv), false, &params);
    ::dynetInitialize(params);
    ::dynetDeleteDynetParams(params);
  }
};

BOOST_FIXTURE_TEST_SUITE(c_exec_test, CExecTest);

BOOST_AUTO_TEST_CASE(autobatch_lstm_gradient) {
  std::vector<float> results;
  ::dynetParameterCollection_t *model;
  ::dynetRNNBuilder_t *builder;
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetCreateParameterCollection(&model));
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetCreateVanillaLSTMBuilder(
                        2, 3, 10, model, false, 1.f, &builder));
  uint32_t dims[] = {3};
  ::dynetLookupParameter_t *lp;
  ::dynetDim_t *dim;
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetCreateDimWithDimensions(dims, 1, &dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetAddLookupParametersToParameterCollection(
                        model, 10, dim, nullptr, nullptr, nullptr, &lp));

  for (size_t i = 0; i < 3; ++i) {
    ::dynetComputationGraph_t *cg;
    BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetCreateComputationGraph(&cg));
    BOOST_CHECK_EQUAL(DYNET_C_OK,
                      ::dynetResetRNNBuilderGraph(builder, cg, true));
    std::vector<::dynetExpression_t*> losses;
    for (size_t j = 0; j < 3; ++j) {
      ::dynetExpression_t *h_0[] = {};
      BOOST_CHECK_EQUAL(DYNET_C_OK,
                        ::dynetStartRNNBuilderNewSequence(
                            builder, h_0, 0));
      for (size_t k = 0; k < 3; ++k) {
        ::dynetExpression_t *x;
        BOOST_CHECK_EQUAL(DYNET_C_OK,
                          ::dynetApplyLookupOne(cg, lp, j * 3 + k, &x));
        ::dynetExpression_t *h;
        BOOST_CHECK_EQUAL(DYNET_C_OK,
                          ::dynetAddRNNBuilderInput(builder, x, &h));
        BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteExpression(h));
        BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteExpression(x));
      }
      std::size_t length = 0u;
      BOOST_CHECK_EQUAL(DYNET_C_OK,
                        ::dynetGetRNNBuilderFinalHiddenState(
                            builder, nullptr, &length));
      ::dynetExpression_t *hs[length];
      BOOST_CHECK_EQUAL(DYNET_C_OK,
                        ::dynetGetRNNBuilderFinalHiddenState(
                            builder, hs, &length));
      ::dynetExpression_t *loss;
      BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetApplySquaredNorm(hs[1], &loss));
      losses.emplace_back(loss);
      for (std::uint32_t t = 0; t < length; ++t) {
        BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteExpression(hs[t]));
      }
    }
    ::dynetExpression_t *loss;
    BOOST_CHECK_EQUAL(DYNET_C_OK,
                      ::dynetApplyAdd(losses[0], losses[2], &loss));
    losses.emplace_back(loss);
    ::dynetExpression_t *z;
    BOOST_CHECK_EQUAL(DYNET_C_OK,
                      ::dynetApplySum(&losses[0], losses.size(), &z));
    const ::dynetTensor_t *z_t;
    BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetGetExpressionValue(z, &z_t));
    float z_v = 0.f;
    BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetEvaluateTensorAsScalar(z_t, &z_v));
    results.emplace_back(z_v);
    BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteExpression(z));
    for (std::uint32_t t = 0; t < losses.size(); ++t) {
      BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteExpression(losses[t]));
    }
    BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteComputationGraph(cg));
  }
  for (size_t i = 1; i < results.size(); ++i) {
    BOOST_CHECK_CLOSE(results[0], results[i], 0.0001);
  }

  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteRNNBuilder(builder));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteDim(dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteLookupParameter(lp));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteParameterCollection(model));
}

BOOST_AUTO_TEST_SUITE_END()
