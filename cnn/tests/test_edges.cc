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
  Tensor U = Ccm({2,3}, {1,2,3,4,5,6});
  Tensor V = Ccm({3,2}, {7,8,9,10,11,12});
  MatrixMultiply mm;
  vector<const Tensor*> xs = {&U, &V};
  Tensor W = mm.forward(xs);
  BOOST_REQUIRE_EQUAL(Dim({2,2}),size(W));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(W(0,0), 58., eps);
  BOOST_CHECK_CLOSE(W(1,0), 139., eps);
  BOOST_CHECK_CLOSE(W(0,1), 64., eps);
  BOOST_CHECK_CLOSE(W(1,1), 154., eps);
}

BOOST_AUTO_TEST_CASE(EColumnConcat)
{
  Tensor u1 = Ccm({2,1}, {1, 4});
  Tensor u2 = Ccm({2,1}, {2, 5});
  Tensor u3 = Ccm({2,1}, {3, 6});
  vector<const Tensor*> xs = {&u1, &u2, &u3};
  ConcatenateColumns cc;
  Tensor U = cc.forward(xs);
  Tensor V = Ccm({3,2}, {7,8,9,10,11,12});
  MatrixMultiply mm;
  vector<const Tensor*> xs2 = {&U, &V};
  Tensor W = mm.forward(xs2);
  BOOST_REQUIRE_EQUAL(Dim({2,2}),size(W));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(W(0,0), 58., eps);
  BOOST_CHECK_CLOSE(W(1,0), 139., eps);
  BOOST_CHECK_CLOSE(W(0,1), 64., eps);
  BOOST_CHECK_CLOSE(W(1,1), 154., eps);
  Tensor b1 = cc.backward(xs, U, U, 0);
  Tensor b2 = cc.backward(xs, U, U, 1);
  Tensor b3 = cc.backward(xs, U, U, 2);
  BOOST_CHECK_EQUAL(u1, b1);
  BOOST_CHECK_EQUAL(u2, b2);
  BOOST_CHECK_EQUAL(u3, b3);
}

BOOST_AUTO_TEST_CASE(ERowConcat)
{
  Tensor u1 = Ccm({2,1}, {1, 4});
  Tensor u2 = Ccm({2,1}, {2, 5});
  Tensor u3 = Ccm({2,1}, {3, 6});
  vector<const Tensor*> xs = {&u1, &u2, &u3};
  Concatenate cr;
  Tensor U = cr.forward(xs);
  BOOST_REQUIRE_EQUAL(Dim({6,1}),size(U));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(U(0,0), 1., eps);
  BOOST_CHECK_CLOSE(U(1,0), 4., eps);
  BOOST_CHECK_CLOSE(U(2,0), 2., eps);
  BOOST_CHECK_CLOSE(U(3,0), 5., eps);
  BOOST_CHECK_CLOSE(U(4,0), 3., eps);
  BOOST_CHECK_CLOSE(U(5,0), 6., eps);

  Tensor b1 = cr.backward(xs, U, U, 0);
  Tensor b2 = cr.backward(xs, U, U, 1);
  Tensor b3 = cr.backward(xs, U, U, 2);
  BOOST_CHECK_EQUAL(u1, b1);
  BOOST_CHECK_EQUAL(u2, b2);
  BOOST_CHECK_EQUAL(u3, b3);
}

BOOST_AUTO_TEST_CASE(ESoftmaxUnif) {
  for (float v = -12.; v < 12.; v += 1.) { 
    Tensor u = Ccm({4,1}, {v, v, v, v});
    Softmax sm;
    vector<const Tensor*> xs = {&u};
    Tensor m = sm.forward(xs);
    BOOST_REQUIRE_EQUAL(Dim({4,1}),size(m));
    double eps = 1e-5;
    for (unsigned i = 0; i < 4; ++i)
      BOOST_CHECK_CLOSE(m(i, 0), 0.25, eps);
    Tensor dEdf = Ccm({4,1}, {1., 0., 0., 0.});
    Tensor d = sm.backward(xs, m, dEdf, 0);
    BOOST_CHECK_CLOSE(d(0,0), 0.1875, eps);
    BOOST_CHECK_CLOSE(d(1,0), -0.0625, eps);
    BOOST_CHECK_CLOSE(d(2,0), -0.0625, eps);
    BOOST_CHECK_CLOSE(d(3,0), -0.0625, eps);

    LogSoftmax lsm;
    Tensor lm = lsm.forward(xs);
    BOOST_REQUIRE_EQUAL(Dim({4,1}),size(lm));
    for (unsigned i = 0; i < 4; ++i)
      BOOST_CHECK_CLOSE(log(m(i, 0)), lm(i, 0), eps);
  }
}

BOOST_AUTO_TEST_CASE(EMultilinear) {
  Tensor b = Ccm({3,1},{1,2,3});
  Tensor W = Ccm({3,2},{2,3,4,5,6,7});
  Tensor x = Ccm({2,1},{-1,1});
  Multilinear ml;
  vector<const Tensor*> mlxs = {&b, &W, &x};
  Tensor r1 = ml.forward(mlxs);
  Sum se;
  MatrixMultiply mm;
  Tensor p = mm.forward(vector<const Tensor*>({&W, &x}));
  Tensor r2 = se.forward(vector<const Tensor*>({&p, &b}));
  BOOST_REQUIRE_EQUAL(size(r1), size(r2));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(r1(0,0), 2., eps);
  BOOST_CHECK_CLOSE(r1(1,0), 3., eps);
  BOOST_CHECK_CLOSE(r1(2,0), 4., eps);
  BOOST_CHECK_CLOSE(r2(0,0), 2., eps);
  BOOST_CHECK_CLOSE(r2(1,0), 3., eps);
  BOOST_CHECK_CLOSE(r2(2,0), 4., eps);
  Tensor dEdf = Ccm({3,1}, {1., 0.5, 0.25});
  Tensor dEdx = ml.backward(mlxs, r1, dEdf, 0);
  BOOST_CHECK_EQUAL(size(dEdx), size(b));
  dEdx = ml.backward(mlxs, r1, dEdf, 1);
  BOOST_CHECK_EQUAL(size(dEdx), size(W));
  dEdx = ml.backward(mlxs, r1, dEdf, 2);
  BOOST_CHECK_EQUAL(size(dEdx), size(x));
}

BOOST_AUTO_TEST_CASE(ELogisticSigmoid) {
  Tensor x = Ccm({5,1},{-6.f,-logf(3),0.f,logf(3),6.f});
  LogisticSigmoid ls;
  vector<const Tensor*> xs = {&x};
  Tensor r = ls.forward(xs);
  BOOST_REQUIRE_EQUAL(size(r), size(x));
  double eps = 1e-2;
  BOOST_CHECK_CLOSE(r(0,0), 1. /(1. + exp(6.)), eps);
  BOOST_CHECK_CLOSE(r(1,0), 0.25, eps);
  BOOST_CHECK_CLOSE(r(2,0), 0.5, eps);
  BOOST_CHECK_CLOSE(r(3,0), 0.75, eps);
  BOOST_CHECK_CLOSE(r(4,0), 1. - r(0,0), eps);
  Tensor dEdf = Ccm({5,1},{1.,1.,1.,1.,1.});
  Tensor dEdx = ls.backward(xs, r, dEdf, 0);
  BOOST_CHECK_CLOSE(dEdx(1,0), 0.1875, eps);
  BOOST_CHECK_CLOSE(dEdx(2,0), 0.25, eps);
  BOOST_CHECK_CLOSE(dEdx(3,0), dEdx(1,0), eps);
  BOOST_CHECK_CLOSE(dEdx(4,0), dEdx(0,0), eps);
}

BOOST_AUTO_TEST_CASE(ETanh) {
  Tensor x = Ccm({5,1},{-6.f,-logf(3),0.f,logf(3),6.f});
  Tanh th;
  vector<const Tensor*> xs = {&x};
  Tensor r = th.forward(xs);
  BOOST_REQUIRE_EQUAL(size(r), size(x));
  double eps = 1e-2;
  BOOST_CHECK_CLOSE(r(1,0), -0.8, eps);
  BOOST_CHECK_CLOSE(r(2,0), 0, eps);
  BOOST_CHECK_CLOSE(r(3,0), 0.8, eps);
  BOOST_CHECK_CLOSE(r(4,0), -r(0,0), eps);
  Tensor dEdf = Ccm({5,1},{1.,1.,1.,1.,1.});
  Tensor dEdx = th.backward(xs, r, dEdf, 0);
  BOOST_CHECK_CLOSE(dEdx(1,0), 0.36, eps);
  BOOST_CHECK_CLOSE(dEdx(2,0), 1.0, eps);
  BOOST_CHECK_CLOSE(dEdx(3,0), dEdx(1,0), eps);
  BOOST_CHECK_CLOSE(dEdx(4,0), dEdx(0,0), eps);
}

