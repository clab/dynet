#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "CNNInit"
#include <boost/test/unit_test.hpp>

#include <vector>

#include "cnn/tensor.h"
#include "cnn/saxe_init.h"
#include "cnn/edges.h"

using namespace std;
using namespace cnn;

BOOST_AUTO_TEST_CASE(EOrthonormalRandom)
{
  for (unsigned d = 3; d < 100; d += 5) {
    Matrix Q = OrthonormalRandom(d, 1.0);
    BOOST_REQUIRE_EQUAL(size(Q), Dim(d,d));

    // check that this is actually returning orthogonal matrices
    Matrix I = Q.transpose() * Q;
    double eps = 1e-1;
    for (unsigned i = 0; i < d; ++i)
      for (unsigned j = 0; j < d; ++j)
        BOOST_CHECK_CLOSE(I(i,j) + 1., (i == j ? 2. : 1.), eps);
  }
}

