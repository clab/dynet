#define BOOST_TEST_MODULE C_DIM_TEST

#include <boost/test/unit_test.hpp>
#include <dynet_c/dim.h>
// #include <test_utils.h>

struct CDimTest {
    CDimTest() {}
};

BOOST_FIXTURE_TEST_SUITE(c_dim_test, CDimTest);

BOOST_AUTO_TEST_CASE(hoge) {
  ::dynetDim_t *dim;
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetCreateDim(&dim));
  BOOST_CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteDim(dim));
}

BOOST_AUTO_TEST_SUITE_END()
