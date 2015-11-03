#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  // parameters
  const unsigned HIDDEN_SIZE = 8;
  const unsigned ITERATIONS = 500;
  Model m;
  SimpleSGDTrainer sgd(&m);
  //MomentumSGDTrainer sgd(&m);

  ComputationGraph cg;

  Expression W = parameter(cg, m.add_parameters({HIDDEN_SIZE, 2}));
  Expression b = parameter(cg, m.add_parameters({HIDDEN_SIZE}));
  Expression V = parameter(cg, m.add_parameters({1, HIDDEN_SIZE}));
  Expression a = parameter(cg, m.add_parameters({1}));

  LookupParameters* x_values = m.add_lookup_parameters(4, {2});
  LookupParameters* y_values = m.add_lookup_parameters(4, {1});
  x_values->Initialize(0, {0.0, 0.0});
  x_values->Initialize(1, {0.0, 1.0});
  x_values->Initialize(2, {1.0, 0.0});
  x_values->Initialize(3, {1.0, 1.0});
  y_values->Initialize(0, {0.0});
  y_values->Initialize(1, {1.0});
  y_values->Initialize(2, {1.0});
  y_values->Initialize(3, {0.0});
  //cerr << "x_values dim is " << x_values->dim << endl;
  //Dim x_dim(std::vector<long>(1,2), 4),
  //Dim y_dim(std::vector<long>(1,1), 4);
  //vector<cnn::real> x_values = {1, 1, -1, 1, 1, -1, -1, -1};  // set x_values to change the inputs to the network
  //vector<cnn::real> y_values = {-1, 1, 1, -1};  // set y_values expressing the output
  //Expression y = input(cg, y_dim, &y_values);

  Expression x = const_batch_lookup(cg, x_values, {0, 1, 2, 3});
  Expression y = const_batch_lookup(cg, y_values, {0, 1, 2, 3});

  cerr << "x is " << x.value().d << ", y is " << y.value().d << endl;
  Expression h = tanh(W*x + b);
  //Expression h = softsign(W*x + b);
  Expression y_pred = V*h + a;
  Expression loss = squared_distance(y_pred, y);

  //cg.PrintGraphviz();
  if (argc == 2) {
    ifstream in(argv[1]);
    boost::archive::text_iarchive ia(in);
    ia >> m;
  }

  // train the parameters
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    vector<float> losses = as_vector(cg.forward());
    cg.backward();
    sgd.update(0.25);
    sgd.update_epoch();
    float loss = 0;
    for(auto l : losses)
      loss += l;
    loss /= 4;
    cerr << "E = " << loss << endl;
  }
  //boost::archive::text_oarchive oa(cout);
  //oa << m;
}

