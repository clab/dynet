#define BOOST_TEST_MODULE C_DIM_TEST

#include <dynet_c/dim.h>
#include <boost/test/unit_test.hpp>
#include "test_utils.h"

struct CDimTest {
    CDimTest() {}
};

BOOST_FIXTURE_TEST_SUITE(c_dim_test, CDimTest);

BOOST_AUTO_TEST_CASE(test_dim_truncate_no_trailing_one) {
  ::dynetDim_t *d1;
  uint32_t dims[] = {1, 3, 4};
  BOOST_CHECK_EQUAL(DYNET_C_OK,
                    ::dynetCreateDimWithDimensions(dims, 3, &d1));
  ::dynetDim_t *t1;
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetTruncateDim(d1, &t1));
  uint32_t nd;
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetGetDimNDimensions(t1, &nd));
  BOOST_CHECK_EQUAL(nd, 3);
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteDim(d1));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteDim(t1));
}

BOOST_AUTO_TEST_SUITE_END()
