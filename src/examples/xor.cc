#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"

#include <iostream>

using namespace std;
using namespace cnn;

int main() {
  srand(time(0));

  RMSPropTrainer sgd;

  // parameters
  const unsigned HIDDEN_SIZE = 8;
  Parameters p_b(Dim(HIDDEN_SIZE,1));
  Parameters p_a(Dim(1,1));
  Parameters p_W(Dim(HIDDEN_SIZE,2));
  Parameters p_V(Dim(1, HIDDEN_SIZE));
  sgd.add_params({&p_b, &p_a, &p_W, &p_V});

  // inputs
  ConstParameters p_x(Dim(2,1));
  ConstParameters p_y(Dim(1,1));

  // build the graph
  Hypergraph hg;
  VariableIndex i_x = hg.add_input(&p_x, "x");
  VariableIndex i_y = hg.add_input(&p_y, "y");
  VariableIndex i_b = hg.add_parameter(&p_b, "b");
  VariableIndex i_a = hg.add_parameter(&p_a, "a");
  VariableIndex i_W = hg.add_parameter(&p_W, "W");
  VariableIndex i_V = hg.add_parameter(&p_V, "V");

  // two options: MatrixMultiply and Sum, or Multilinear
#if 0
  VariableIndex i_f = hg.add_function<MatrixMultiply>({i_W, i_x}, "f");
  VariableIndex i_g = hg.add_function<Sum>({i_f, i_b}, "g");
#else
  VariableIndex i_g = hg.add_function<Multilinear>({i_b, i_W, i_x}, "g");
#endif
  VariableIndex i_h = hg.add_function<Tanh>({i_g}, "h");

#if 0
  VariableIndex i_p = hg.add_function<MatrixMultiply>({i_V, i_h}, "p");
  VariableIndex i_y_pred = hg.add_function<Sum>({i_p, i_a}, "y_pred");
#else
  VariableIndex i_y_pred = hg.add_function<Multilinear>({i_a, i_V, i_h}, "y_pred");
#endif
  hg.add_function<SquaredEuclideanDistance>({i_y_pred, i_y}, "err");
  hg.PrintGraphviz();

  // train the parameters
  for (unsigned iter = 0; iter < 30; ++iter) {
    double loss = 0;
    for (unsigned mi = 0; mi < 4; ++mi) {
      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      p_x(0,0) = x1 ? 1 : -1;
      p_x(1,0) = x2 ? 1 : -1;
      p_y(0,0) = (x1 != x2) ? 1 : -1;
      loss += hg.forward()(0,0);
      hg.backward();
      sgd.update(1.0);
    }
    loss /= 4;
    cerr << "E = " << loss << endl;
  }
}

