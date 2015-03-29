#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "CNNInit"
#include <boost/test/unit_test.hpp>

#include <vector>

#include "cnn/tests/test_utils.h"
#include "cnn/tensor.h"
#include "cnn/saxe_init.h"

using namespace std;
using namespace cnn;

BOOST_GLOBAL_FIXTURE(TestTensorSetup)

BOOST_AUTO_TEST_CASE(EOrthonormalRandom)
{
  for (int d = 4; d < 100; d += 16) {
    Tensor Q = OrthonormalRandom(d, 1.0);
    BOOST_REQUIRE_EQUAL(size(Q), Dim({d,d}));

    // check that this is actually returning orthogonal matrices
#if MINERVA_BACKEND
    Tensor I = Q.Trans() * Q;
#else
    Tensor I = Q.transpose() * Q;
#endif
    double eps = 1e-1;
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < d; ++j)
        BOOST_CHECK_CLOSE(t(I,i,j) + 1., (i == j ? 2. : 1.), eps);
  }
}

BOOST_AUTO_TEST_CASE(BernoulliInit) {
  Tensor r = RandomBernoulli(Dim({1000,1000}), 0.5f);
  int tot = 0;
  for (int i = 0; i < 1000; ++i)
    for (int j = 0; j < 1000; ++j)
      if (t(r,i,j)) ++tot;
  cerr << "tot: " << tot << endl;
  BOOST_CHECK_GT(tot, 490000);
  BOOST_CHECK_LT(tot, 510000);
}

