#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "CNNEdges"
#include <boost/test/unit_test.hpp>

#include <vector>

#include "cnn/tests/test_utils.h"
#include "cnn/tensor.h"
#include "cnn/edges.h"
#include "cnn/c2w.h"

using namespace std;
using namespace cnn;

BOOST_GLOBAL_FIXTURE(TestTensorSetup)

Dim size(const Tensor& t) {
  if (t.cols() > 1)
    return Dim(t.rows(), t.cols());
  return Dim(t.rows());
}

BOOST_AUTO_TEST_CASE(ESqrL2)
{
  auto U = Ccm({2}, {4,5});
  auto V = Ccm({2}, {1,1});
  cerr << str(U) << endl;
  SquaredEuclideanDistance e;
  vector<const Tensor*> xs = {&U, &V};
  Tensor W = e.forward(xs); 
  cerr << "Norm^2:" << str(W) << endl;
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(t(W,0),25., eps);
  Tensor dEdf = Ccm({1}, {1});
  Tensor d1 = e.backward(xs, W, dEdf, 0);
  Tensor d2 = e.backward(xs, W, dEdf, 1);
  cerr << d1 << endl;
  cerr << d2 << endl;
  BOOST_CHECK_CLOSE(t(d1,0), 6., eps);
  BOOST_CHECK_CLOSE(t(d1,1), 8., eps);
  BOOST_CHECK_CLOSE(t(d2,0), -6., eps);
  BOOST_CHECK_CLOSE(t(d2,1), -8., eps);
}

BOOST_AUTO_TEST_CASE(EMatrixMultiply) {
  Tensor U = Ccm({2,3}, {1,2,3,4,5,6});
  Tensor V = Ccm({3,2}, {7,8,9,10,11,12});
  MatrixMultiply mm;
  vector<const Tensor*> xs = {&U, &V};
  Tensor W = mm.forward(xs);
  BOOST_REQUIRE_EQUAL(Dim({2,2}),size(W));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(t(W,0,0), 76., eps);
  BOOST_CHECK_CLOSE(t(W,1,0), 100., eps);
  BOOST_CHECK_CLOSE(t(W,0,1), 103., eps);
  BOOST_CHECK_CLOSE(t(W,1,1), 136., eps);
  Tensor dEdf = Ccm({2,2}, {-1,0.5,1,2});
  Tensor dEdx0 = mm.backward(xs, W, dEdf, 0);
  cerr << str(dEdx0) << endl;
  BOOST_CHECK_CLOSE(t(dEdx0,0,0),3.,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,0,1),3.,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,0,2),3.,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,1,0),23.5,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,1,1),26.,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,1,2),28.5,eps);
  Tensor dEdx1 = mm.backward(xs, W, dEdf, 1);
  cerr << str(dEdx1) << endl;
  BOOST_CHECK_CLOSE(t(dEdx1,0,0),0.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,1,0),-1.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,2,0),-2.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,0,1),5.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,1,1),11.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,2,1),17.,eps);
}

BOOST_AUTO_TEST_CASE(EColumnConcat)
{
  Tensor u1 = Ccm({2}, {1, 2});
  Tensor u2 = Ccm({2}, {3, 4});
  Tensor u3 = Ccm({2}, {5, 6});
  cerr << u1 << endl;
  cerr << u2 << endl;
  cerr << u3 << endl;
  vector<const Tensor*> xs = {&u1, &u2, &u3};
  ConcatenateColumns cc;
  Tensor U = cc.forward(xs);
  cerr << U << endl;
  Tensor V = Ccm({3,2}, {7,8,9,10,11,12});
  MatrixMultiply mm;
  vector<const Tensor*> xs2 = {&U, &V};
  Tensor W = mm.forward(xs2);
  cerr << W << endl;
  cerr << str(W) << endl;
  BOOST_REQUIRE_EQUAL(Dim({2,2}),size(W));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(t(W,0,0), 76., eps);
  BOOST_CHECK_CLOSE(t(W,1,0), 100., eps);
  BOOST_CHECK_CLOSE(t(W,0,1), 103., eps);
  BOOST_CHECK_CLOSE(t(W,1,1), 136., eps);
  Tensor b1 = cc.backward(xs, U, U, 0);
  Tensor b2 = cc.backward(xs, U, U, 1);
  Tensor b3 = cc.backward(xs, U, U, 2);
  cerr << str(b1) << endl;
  cerr << str(b2) << endl;
  cerr << str(b3) << endl;
  BOOST_CHECK_EQUAL(t(u1,0), t(b1,0));
  BOOST_CHECK_EQUAL(t(u1,1), t(b1,1));
  BOOST_CHECK_EQUAL(t(u2,0), t(b2,0));
  BOOST_CHECK_EQUAL(t(u2,1), t(b2,1));
  BOOST_CHECK_EQUAL(t(u3,0), t(b3,0));
  BOOST_CHECK_EQUAL(t(u3,1), t(b3,1));
}

BOOST_AUTO_TEST_CASE(ERowConcat)
{
  Tensor u1 = Ccm({2}, {1, 4});
  Tensor u2 = Ccm({2}, {2, 5});
  Tensor u3 = Ccm({3}, {3, 6, 7});
  cerr << str(u1) << endl;
  cerr << str(u2) << endl;
  cerr << str(u3) << endl;
  vector<const Tensor*> xs = {&u1, &u2, &u3};
  Concatenate cr;
  Tensor U = cr.forward(xs);
  cerr << str(U) << endl;
  //BOOST_REQUIRE_EQUAL(Dim({7}),size(U));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(t(U,0), 1., eps);
  BOOST_CHECK_CLOSE(t(U,1), 4., eps);
  BOOST_CHECK_CLOSE(t(U,2), 2., eps);
  BOOST_CHECK_CLOSE(t(U,3), 5., eps);
  BOOST_CHECK_CLOSE(t(U,4), 3., eps);
  BOOST_CHECK_CLOSE(t(U,5), 6., eps);

  Tensor b1 = cr.backward(xs, U, U, 0);
  Tensor b2 = cr.backward(xs, U, U, 1);
  Tensor b3 = cr.backward(xs, U, U, 2);
  cerr << str(b1) << endl;
  cerr << str(b2) << endl;
  cerr << str(b3) << endl;
  BOOST_CHECK_EQUAL(t(u1,0), t(b1,0));
  BOOST_CHECK_EQUAL(t(u1,1), t(b1,1));
  BOOST_CHECK_EQUAL(t(u2,0), t(b2,0));
  BOOST_CHECK_EQUAL(t(u2,1), t(b2,1));
  BOOST_CHECK_EQUAL(t(u3,0), t(b3,0));
  BOOST_CHECK_EQUAL(t(u3,1), t(b3,1));
  BOOST_CHECK_EQUAL(t(u3,2), t(b3,2));
}

BOOST_AUTO_TEST_CASE(EMultilinear) {
  Tensor b = Ccm({3},{1,2,3});
  Tensor W = Ccm({3,2},{2,4,6,3,5,7});
  Tensor x = Ccm({2},{-1,1});
  Multilinear ml;
  vector<const Tensor*> mlxs = {&b, &W, &x};
  Tensor r1 = ml.forward(mlxs);
  Sum se;
  MatrixMultiply mm;
  Tensor p = mm.forward(vector<const Tensor*>({&W, &x}));
  Tensor r2 = se.forward(vector<const Tensor*>({&p, &b}));
  BOOST_REQUIRE(size(r1) == size(r2));
  double eps = 1e-5;
  cerr << r1 << endl;
  cerr << r2 << endl;
  BOOST_CHECK_CLOSE(t(r1,0), 2., eps);
  BOOST_CHECK_CLOSE(t(r1,1), 3., eps);
  BOOST_CHECK_CLOSE(t(r1,2), 4., eps);
  BOOST_CHECK_CLOSE(t(r2,0), 2., eps);
  BOOST_CHECK_CLOSE(t(r2,1), 3., eps);
  BOOST_CHECK_CLOSE(t(r2,2), 4., eps);
  cerr << "Multilinear forward complete\n";
  Tensor dEdf = Ccm({3}, {1., 0.5, 0.25});
  Tensor dEdx = ml.backward(mlxs, r1, dEdf, 0);
  cerr << "BACK 0:\n";
  BOOST_CHECK(size(dEdx) == size(b));
  cerr << str(dEdx) << endl;
  BOOST_CHECK_CLOSE(t(dEdx,0), 1., eps);
  BOOST_CHECK_CLOSE(t(dEdx,1), 0.5, eps);
  BOOST_CHECK_CLOSE(t(dEdx,2), 0.25, eps);
  dEdx = ml.backward(mlxs, r1, dEdf, 1);
  cerr << "BACK 1:\n";
  BOOST_CHECK(size(dEdx) == size(W));
  cerr << str(dEdx) << endl;
  BOOST_CHECK_CLOSE(t(dEdx,0,0), -1., eps);
  BOOST_CHECK_CLOSE(t(dEdx,1,0), -0.5, eps);
  BOOST_CHECK_CLOSE(t(dEdx,2,0), -0.25, eps);
  BOOST_CHECK_CLOSE(t(dEdx,0,1), 1., eps);
  BOOST_CHECK_CLOSE(t(dEdx,1,1), 0.5, eps);
  BOOST_CHECK_CLOSE(t(dEdx,2,1), 0.25, eps);
  dEdx = ml.backward(mlxs, r1, dEdf, 2);
  cerr << "BACK 2:\n";
  BOOST_CHECK(size(dEdx) == size(x));
  cerr << str(dEdx) << endl;
  BOOST_CHECK_CLOSE(t(dEdx,0), 5.5, eps);
  BOOST_CHECK_CLOSE(t(dEdx,1), 7.25, eps);
}

BOOST_AUTO_TEST_CASE(ELogisticSigmoid) {
  Tensor x = Ccm({5,1},{-6.f,-logf(3),0.f,logf(3),6.f});
  LogisticSigmoid ls;
  vector<const Tensor*> xs = {&x};
  Tensor r = ls.forward(xs);
  BOOST_REQUIRE_EQUAL(size(r), size(x));
  double eps = 1e-2;
  BOOST_CHECK_CLOSE(t(r,0,0), 1. /(1. + exp(6.)), eps);
  BOOST_CHECK_CLOSE(t(r,1,0), 0.25, eps);
  BOOST_CHECK_CLOSE(t(r,2,0), 0.5, eps);
  BOOST_CHECK_CLOSE(t(r,3,0), 0.75, eps);
  BOOST_CHECK_CLOSE(t(r,4,0), 1. - t(r,0,0), eps);
  cerr << "HERE\n";
  cerr << str(r) << endl;
  Tensor dEdf = Ccm({5,1},{1.,1.,1.,1.,1.});
  Tensor dEdx = ls.backward(xs, r, dEdf, 0);
  BOOST_CHECK_CLOSE(t(dEdx,1,0), 0.1875, eps);
  BOOST_CHECK_CLOSE(t(dEdx,2,0), 0.25, eps);
  BOOST_CHECK_CLOSE(t(dEdx,3,0), t(dEdx,1,0), eps);
  BOOST_CHECK_CLOSE(t(dEdx,4,0), t(dEdx,0,0), eps);
}

BOOST_AUTO_TEST_CASE(ETanh) {
  Tensor x = Ccm({5,1},{-6.f,-logf(3),0.f,logf(3),6.f});
  Tanh th;
  vector<const Tensor*> xs = {&x};
  Tensor r = th.forward(xs);
  BOOST_REQUIRE(size(r) == size(x));
  double eps = 1e-2;
  BOOST_CHECK_CLOSE(t(r,1,0), -0.8, eps);
  BOOST_CHECK_CLOSE(t(r,2,0), 0, eps);
  BOOST_CHECK_CLOSE(t(r,3,0), 0.8, eps);
  BOOST_CHECK_CLOSE(t(r,4,0), -t(r,0,0), eps);
  Tensor dEdf = Ccm({5,1},{1.,1.,1.,1.,1.});
  cerr << "Tanh complete forward\n";
  Tensor dEdx = th.backward(xs, r, dEdf, 0);
  BOOST_CHECK_CLOSE(t(dEdx,1,0), 0.36, eps);
  BOOST_CHECK_CLOSE(t(dEdx,2,0), 1.0, eps);
  BOOST_CHECK_CLOSE(t(dEdx,3,0), t(dEdx,1,0), eps);
  BOOST_CHECK_CLOSE(t(dEdx,4,0), t(dEdx,0,0), eps);
}

BOOST_AUTO_TEST_CASE(MatrixVector) {
  cerr << "Matrix-Vector\n";
  Tensor W = Ccm({3,2},{2,4,6,3,5,7});
  Tensor x = Ccm({2},{-1,1});
  MatrixMultiply mm;
  vector<const Tensor*> xs = {&W, &x};
  Tensor fx = mm.forward(xs);
  cerr << str(fx) << endl;
  Tensor dEdf = Ccm({3},{-.5,0.25,5});
  Tensor M = mm.backward(xs, fx, dEdf, 0);
  cerr << "Diff with respect to W:\n";
  cerr << str(M) << endl;
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(t(M,0,0), 0.5, eps);
  BOOST_CHECK_CLOSE(t(M,1,0), -0.25, eps);
  BOOST_CHECK_CLOSE(t(M,2,0), -5, eps);
  BOOST_CHECK_CLOSE(t(M,0,1), -0.5, eps);
  BOOST_CHECK_CLOSE(t(M,1,1), 0.25, eps);
  BOOST_CHECK_CLOSE(t(M,2,1), 5, eps);
  Tensor vv = mm.backward(xs, fx, dEdf, 1);
  cerr << "Diff with respect to x:\n";
  cerr << str(vv) << endl;
  BOOST_CHECK_CLOSE(t(vv,0), 30., eps);
  BOOST_CHECK_CLOSE(t(vv,1), 34.75, eps);
}

BOOST_AUTO_TEST_CASE(EConstantMinus) {
  Tensor W = Ccm({2,2},{1,2,3,-4});
  ConstantMinusX om(1);
  vector<const Tensor*> xs(1, &W);
  Tensor O = om.forward(xs);
  cerr << str(W) << endl;
  cerr << str(O) << endl;
  double eps = 1e-6;
  BOOST_CHECK_CLOSE(10 + 1 - t(W,0,0), 10 + t(O,0,0), eps);
  BOOST_CHECK_CLOSE(10 + 1 - t(W,0,1), 10 + t(O,0,1), eps);
  BOOST_CHECK_CLOSE(10 + 1 - t(W,1,0), 10 + t(O,1,0), eps);
  BOOST_CHECK_CLOSE(10 + 1 - t(W,1,1), 10 + t(O,1,1), eps);
  Tensor V = -W;
  cerr << str(W) << endl;
  cerr << str(V) << endl;
}

BOOST_AUTO_TEST_CASE(ESoftmaxUnif) {
  for (float v = -12.; v < 12.; v += 1.) { 
    Tensor u = Ccm({4}, {v, v, v, v});
    Softmax sm;
    vector<const Tensor*> xs = {&u};
    Tensor m = sm.forward(xs);
    BOOST_REQUIRE_EQUAL(Dim({4}),size(m));
    double eps = 1e-5;
    for (unsigned i = 0; i < 4; ++i)
      BOOST_CHECK_CLOSE(t(m, i), 0.25, eps);
    Tensor dEdf = Ccm({4}, {1., 0., 0., 0.});
    Tensor d = sm.backward(xs, m, dEdf, 0);
    BOOST_CHECK_CLOSE(t(d,0), 0.1875, eps);
    BOOST_CHECK_CLOSE(t(d,1), -0.0625, eps);
    BOOST_CHECK_CLOSE(t(d,2), -0.0625, eps);
    BOOST_CHECK_CLOSE(t(d,3), -0.0625, eps);
//    cerr << d << endl;

    LogSoftmax lsm;
    Tensor lm = lsm.forward(xs);
    BOOST_REQUIRE_EQUAL(Dim({4}),size(lm));
    for (unsigned i = 0; i < 4; ++i)
      BOOST_CHECK_CLOSE(log(t(m, i)), t(lm, i), eps);
    Tensor b = lsm.backward(xs, lm, dEdf, 0);
    BOOST_CHECK_CLOSE(t(b, 0), 0.75, eps);
    BOOST_CHECK_CLOSE(t(b, 1), -0.25, eps);
    BOOST_CHECK_CLOSE(t(b, 2), -0.25, eps);
    BOOST_CHECK_CLOSE(t(b, 3), -0.25, eps);
  }
}

#ifdef WITH_THPP_BACKEND
BOOST_AUTO_TEST_CASE(TensorInner3D_1D) {
  Tensor A = Ccm({24}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  A.reshape(A, {2,3,4});
  Tensor v = Ccm({4}, {-0.5, 1, 1.5, 2});
  Tensor B = Ccm({2,3}, {1, 2, 3, 4, 5, 6});
  vector<const Tensor*> xs = {&A, &v, &B};
  InnerProduct3D_1D e;
  Tensor Y = e.forward(xs);
  cerr << str(Y) << endl;
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(t(Y, 0, 0), 11, eps);
  BOOST_CHECK_CLOSE(t(Y, 1, 0), 60, eps);
  BOOST_CHECK_CLOSE(t(Y, 0, 1), 29, eps);
  BOOST_CHECK_CLOSE(t(Y, 1, 1), 78, eps);
  BOOST_CHECK_CLOSE(t(Y, 0, 2), 47, eps);
  BOOST_CHECK_CLOSE(t(Y, 1, 2), 96, eps);
  Tensor dEdY = Ccm({2,3}, {1, 0.1, -1, 1.2, 2, -0.25});
  Tensor dEdx3 = e.backward(xs, Y, dEdY, 2);
  cerr << str(dEdY) << endl;
  cerr << str(dEdx3) << endl;
  Tensor dEdx1 = e.backward(xs, Y, dEdY, 0);
  cerr << dEdx1 << endl;
  //cerr << str(dEdx1) << endl;
  Tensor dEdx2 = e.backward(xs, Y, dEdY, 1);
  cerr << str(dEdx2) << endl;
}
#endif

