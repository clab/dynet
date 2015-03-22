#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "CNNEdges"
#include <boost/test/unit_test.hpp>

#include <vector>

#include "cnn/tensor.h"
#include "cnn/edges.h"

using namespace std;
using namespace cnn;

BOOST_AUTO_TEST_CASE(EMatrixMultiply)
{
  Matrix U = Ccm({2,3}, {1,2,3,4,5,6});
  Matrix V = Ccm({3,2}, {7,8,9,10,11,12});
  MatrixMultiply mm;
  vector<const Matrix*> xs = {&U, &V};
  Matrix W = mm.forward(xs);
  BOOST_REQUIRE_EQUAL(Dim({2,2}),size(W));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(W(0,0), 58., eps);
  BOOST_CHECK_CLOSE(W(1,0), 139., eps);
  BOOST_CHECK_CLOSE(W(0,1), 64., eps);
  BOOST_CHECK_CLOSE(W(1,1), 154., eps);
}

BOOST_AUTO_TEST_CASE(EColumnConcat)
{
  Matrix u1 = Ccm({2,1}, {1, 4});
  Matrix u2 = Ccm({2,1}, {2, 5});
  Matrix u3 = Ccm({2,1}, {3, 6});
  vector<const Matrix*> xs = {&u1, &u2, &u3};
  ConcatenateColumns cc;
  Matrix U = cc.forward(xs);
  Matrix V = Ccm({3,2}, {7,8,9,10,11,12});
  MatrixMultiply mm;
  vector<const Matrix*> xs2 = {&U, &V};
  Matrix W = mm.forward(xs2);
  BOOST_REQUIRE_EQUAL(Dim({2,2}),size(W));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(W(0,0), 58., eps);
  BOOST_CHECK_CLOSE(W(1,0), 139., eps);
  BOOST_CHECK_CLOSE(W(0,1), 64., eps);
  BOOST_CHECK_CLOSE(W(1,1), 154., eps);
  Matrix b1 = cc.backward(xs, U, U, 0);
  Matrix b2 = cc.backward(xs, U, U, 1);
  Matrix b3 = cc.backward(xs, U, U, 2);
  BOOST_CHECK_EQUAL(u1, b1);
  BOOST_CHECK_EQUAL(u2, b2);
  BOOST_CHECK_EQUAL(u3, b3);
}

BOOST_AUTO_TEST_CASE(ERowConcat)
{
  Matrix u1 = Ccm({2,1}, {1, 4});
  Matrix u2 = Ccm({2,1}, {2, 5});
  Matrix u3 = Ccm({2,1}, {3, 6});
  vector<const Matrix*> xs = {&u1, &u2, &u3};
  Concatenate cr;
  Matrix U = cr.forward(xs);
  BOOST_REQUIRE_EQUAL(Dim({6,1}),size(U));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(U(0,0), 1., eps);
  BOOST_CHECK_CLOSE(U(1,0), 4., eps);
  BOOST_CHECK_CLOSE(U(2,0), 2., eps);
  BOOST_CHECK_CLOSE(U(3,0), 5., eps);
  BOOST_CHECK_CLOSE(U(4,0), 3., eps);
  BOOST_CHECK_CLOSE(U(5,0), 6., eps);

  Matrix b1 = cr.backward(xs, U, U, 0);
  Matrix b2 = cr.backward(xs, U, U, 1);
  Matrix b3 = cr.backward(xs, U, U, 2);
  BOOST_CHECK_EQUAL(u1, b1);
  BOOST_CHECK_EQUAL(u2, b2);
  BOOST_CHECK_EQUAL(u3, b3);
}

BOOST_AUTO_TEST_CASE(ESoftmaxUnif) {
  for (float v = -12.; v < 12.; v += 1.) { 
    Matrix u = Ccm({4,1}, {v, v, v, v});
    Softmax sm;
    vector<const Matrix*> xs = {&u};
    Matrix m = sm.forward(xs);
    BOOST_REQUIRE_EQUAL(Dim({4,1}),size(m));
    double eps = 1e-5;
    for (unsigned i = 0; i < 4; ++i)
      BOOST_CHECK_CLOSE(m(i, 0), 0.25, eps);
    Matrix dEdf = Ccm({4,1}, {1., 0., 0., 0.});
    Matrix d = sm.backward(xs, m, dEdf, 0);
    BOOST_CHECK_CLOSE(d(0,0), 0.1875, eps);
    BOOST_CHECK_CLOSE(d(1,0), -0.0625, eps);
    BOOST_CHECK_CLOSE(d(2,0), -0.0625, eps);
    BOOST_CHECK_CLOSE(d(3,0), -0.0625, eps);

    LogSoftmax lsm;
    Matrix lm = lsm.forward(xs);
    BOOST_REQUIRE_EQUAL(Dim({4,1}),size(lm));
    for (unsigned i = 0; i < 4; ++i)
      BOOST_CHECK_CLOSE(log(m(i, 0)), lm(i, 0), eps);
  }
}

