#define BOOST_TEST_MODULE TEST_DIM

#include <dynet/dim.h>

#include "test.h"

using namespace dynet;

struct DimTest {

};

BOOST_FIXTURE_TEST_SUITE(dim_test, DimTest);

BOOST_AUTO_TEST_CASE( test_dim_truncate_no_trailing_one ) {
    Dim d1({1,3,4});
    Dim t1 = d1.truncate();
    BOOST_CHECK_EQUAL(t1.nd, 3);
}

BOOST_AUTO_TEST_CASE( test_dim_truncate_all_one ) {
    Dim d1({1,1,1});
    Dim t1 = d1.truncate();
    BOOST_CHECK_EQUAL(t1.nd, 1);
}

BOOST_AUTO_TEST_CASE( test_dim_truncate_trailing_one ) {
    Dim d1({4,3,1});
    Dim t1 = d1.truncate();
    BOOST_CHECK_EQUAL(t1.nd, 2);
}

BOOST_AUTO_TEST_CASE( test_dim_truncate_multiple_one ) {
    Dim d1({4,1,1});
    Dim t1 = d1.truncate();
    BOOST_CHECK_EQUAL(t1.nd, 1);
}

BOOST_AUTO_TEST_CASE( test_dim_truncate_a_one ) {
    Dim d1({1});
    Dim t1 = d1.truncate();
    BOOST_CHECK_EQUAL(t1.nd, 1);
}

BOOST_AUTO_TEST_SUITE_END()

