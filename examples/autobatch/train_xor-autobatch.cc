#include "dynet/training.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  // parameters
  const unsigned ITERATIONS = 30;
  ParameterCollection m;
  SimpleSGDTrainer trainer(m);
  //MomentumSGDTrainer trainer(m);

  Parameter p_W, p_b, p_V, p_a;
  const unsigned HIDDEN_SIZE = 3;
  p_W = m.add_parameters({HIDDEN_SIZE, 2});
  p_b = m.add_parameters({HIDDEN_SIZE});
  p_V = m.add_parameters({1, HIDDEN_SIZE});
  p_a = m.add_parameters({1});

  // train the parameters
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {

    ComputationGraph cg;
    Expression W = parameter(cg, p_W);
    Expression b = parameter(cg, p_b);
    Expression V = parameter(cg, p_V);
    Expression a = parameter(cg, p_a);

    vector<Expression> losses;

    for (unsigned mi = 0; mi < 4; ++mi) {

      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      vector<dynet::real> x_values(2);
      x_values[0] = x1 ? 1 : -1;
      x_values[1] = x2 ? 1 : -1;
      float y_value = (x1 != x2) ? 1 : -1;

      Expression x = input(cg, {2}, x_values);
      Expression y = input(cg, y_value);

      //Expression h = tanh(W*x + b);
      Expression h = tanh(affine_transform({b, W, x}));
      //Expression h = softsign(W*x + b);
      Expression y_pred = affine_transform({a, V, h});
      losses.push_back(squared_distance(y_pred, y));

    }

    Expression loss_expr = sum(losses);

    // Print the graph, just for fun.
    if(iter == 0) {
      cg.print_graphviz();
    }

    // Calculate the loss. Batching will automatically be done here.
    float loss = as_scalar(cg.forward(loss_expr)) / 4;
    cg.backward(loss_expr);
    trainer.update();

    cerr << "E = " << loss << endl;
  }

}

