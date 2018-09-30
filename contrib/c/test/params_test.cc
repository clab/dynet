#define BOOST_TEST_MODULE C_PARAMS_TEST

#include <dynet_c/model.h>
#include <boost/test/unit_test.hpp>
#include "test_utils.h"

struct CParamsTest {
  CParamsTest() {
    test_utils::init_dynet();
  }
};

BOOST_FIXTURE_TEST_SUITE(c_params_test, CParamsTest);

BOOST_AUTO_TEST_CASE(test_parameter_collection) {
  ::dynetParameterCollection_t *model;
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetCreateParameterCollection(&model));
  ::dynetParameter_t *a;
  ::dynetDim_t *a_dim;
  uint32_t a_dims[] = {10};
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetCreateDimWithDimensions(a_dims, 1, &a_dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetAddParametersToParameterCollection(
                        model, a_dim, nullptr, nullptr, nullptr, &a));
  ::dynetParameter_t *b;
  ::dynetDim_t *b_dim;
  uint32_t b_dims[] = {1, 2};
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetCreateDimWithDimensions(b_dims, 2, &b_dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetAddParametersToParameterCollection(
                        model, b_dim, nullptr, nullptr, nullptr, &b));
  ::dynetParameterCollection_t *submodel;
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetAddSubcollectionToParameterCollection(
                        model, "foo", &submodel));
  ::dynetParameter_t *c;
  ::dynetDim_t *c_dim;
  uint32_t c_dims[] = {10};
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetCreateDimWithDimensions(c_dims, 1, &c_dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetAddParametersToParameterCollection(
                        submodel, c_dim, nullptr, nullptr, nullptr, &c));
  ::dynetParameter_t *d;
  ::dynetDim_t *d_dim;
  uint32_t d_dims[] = {1, 2};
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetCreateDimWithDimensions(d_dims, 2, &d_dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetAddParametersToParameterCollection(
                        submodel, d_dim, nullptr, nullptr, nullptr, &d));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteDim(a_dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteParameter(a));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteDim(b_dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteParameter(b));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteDim(c_dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteParameter(c));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteDim(d_dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteParameter(d));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteParameterCollection(submodel));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteParameterCollection(model));
}

BOOST_AUTO_TEST_SUITE_END()
