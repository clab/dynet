#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  // parameters
  const unsigned HIDDEN_SIZE = 8;
  const unsigned ITERATIONS = 200;
  ParameterCollection m;
  SimpleSGDTrainer trainer(m);

  ComputationGraph cg;
  Parameter p_W, p_b, p_V, p_a;

  p_W = m.add_parameters({HIDDEN_SIZE, 2});
  p_b = m.add_parameters({HIDDEN_SIZE});
  p_V = m.add_parameters({1, HIDDEN_SIZE});
  p_a = m.add_parameters({1});

  if (argc == 2) {
    // Load the model and parameters from file if given.
    TextFileLoader loader(argv[1]);
    loader.populate(m);
  }

  Expression W = parameter(cg, p_W);
  Expression b = parameter(cg, p_b);
  Expression V = parameter(cg, p_V);
  Expression a = parameter(cg, p_a);

  // set x_values to change the inputs to the network
  Dim x_dim({2}, 4), y_dim({1}, 4);
  cerr << "x_dim=" << x_dim << ", y_dim=" << y_dim << endl;
  vector<dynet::real> x_values = {1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0};
  Expression x = input(cg, x_dim, &x_values);
  // set y_values expressing the output
  vector<dynet::real> y_values = {-1.0, 1.0, 1.0, -1.0};
  Expression y = input(cg, y_dim, &y_values);

  Expression h = tanh(W*x + b);
  //Expression h = tanh(affine_transform({b, W, x}));
  //Expression h = softsign(W*x + b);
  Expression y_pred = V*h + a;
  Expression loss = squared_distance(y_pred, y);
  Expression sum_loss = sum_batches(loss);

  // train the parameters
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    float my_loss = as_scalar(cg.forward(sum_loss)) / 4;
    cg.backward(sum_loss);
    trainer.update();
    cerr << "E = " << my_loss << endl;
  }

  // Output the model and parameter objects
  // to a cout.
  TextFileSaver saver("xor-batch.model");
  saver.save(m);
}
