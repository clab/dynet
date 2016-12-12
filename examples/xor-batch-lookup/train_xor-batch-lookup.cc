#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  // parameters
  const unsigned HIDDEN_SIZE = 8;
  const unsigned ITERATIONS = 200;
  Model m;
  SimpleSGDTrainer sgd(m);

  ComputationGraph cg;
  Parameter p_W, p_b, p_V, p_a;
  LookupParameter x_values, y_values;
  if (argc == 2) {
    // Load the model and parameters from
    // file if given.
    ifstream in(argv[1]);
    boost::archive::text_iarchive ia(in);
    ia >> m >> p_W >> p_b >> p_V >> p_a;
  }
  else {
    // Otherwise, just create a new model.
    p_W = m.add_parameters({HIDDEN_SIZE, 2});
    p_b = m.add_parameters({HIDDEN_SIZE});
    p_V = m.add_parameters({1, HIDDEN_SIZE});
    p_a = m.add_parameters({1});

    x_values = m.add_lookup_parameters(4, {2});
    y_values = m.add_lookup_parameters(4, {1});
    x_values.initialize(0, {1.0, 1.0});
    x_values.initialize(1, {-1.0, 1.0});
    x_values.initialize(2, {1.0, -1.0});
    x_values.initialize(3, {-1.0, -1.0});
    y_values.initialize(0, {-1.0});
    y_values.initialize(1, {1.0});
    y_values.initialize(2, {1.0});
    y_values.initialize(3, {-1.0});
  }

  Expression W = parameter(cg, p_W);
  Expression b = parameter(cg, p_b);
  Expression V = parameter(cg, p_V);
  Expression a = parameter(cg, p_a);

  Expression x = const_lookup(cg, x_values, {0, 1, 2, 3});
  Expression y = const_lookup(cg, y_values, {0, 1, 2, 3});

  cerr << "x is " << x.value().d << ", y is " << y.value().d << endl;
  Expression h = tanh(W*x + b);
  //Expression h = softsign(W*x + b);
  Expression y_pred = V*h + a;
  Expression loss = squared_distance(y_pred, y);
  Expression sum_loss = sum_batches(loss);

  cg.print_graphviz();

  // train the parameters
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    vector<float> losses = as_vector(cg.forward(sum_loss));
    cg.backward(sum_loss);
    sgd.update(0.25);
    sgd.update_epoch();
    float loss = 0;
    for(auto l : losses)
      loss += l;
    loss /= 4;
    cerr << "E = " << loss << endl;
  }
  // Output the model and parameter objects
  // to a cout.
  boost::archive::text_oarchive oa(cout);
  oa << m << p_W << p_b << p_V << p_a << x_values << y_values;
}

