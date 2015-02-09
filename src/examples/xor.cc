#include "cnn/edges.h"
#include "cnn/cnn.h"

#include <iostream>

using namespace std;
using namespace cnn;

int main() {
  sranddev();

  // build the graph
  Hypergraph hg;
  const unsigned HIDDEN_SIZE = 6;
  ConstParameters p_x(Dim(2,1));
  unsigned i_x = hg.add_input(&p_x, "x");
  ConstParameters p_y(Dim(1,1));
  unsigned i_y = hg.add_input(&p_y, "y");
  Parameters p_b(Dim(HIDDEN_SIZE,1));
  unsigned i_b = hg.add_parameter(&p_b, "b");
  Parameters p_a(Dim(1,1));
  unsigned i_a = hg.add_parameter(&p_a, "a");
  Parameters p_W(Dim(HIDDEN_SIZE,2));
  unsigned i_W = hg.add_parameter(&p_W, "W");
  Parameters p_V(Dim(1, HIDDEN_SIZE));
  unsigned i_V = hg.add_parameter(&p_V, "V");
  unsigned i_f = hg.add_function<MatrixMultiply>({i_W, i_x}, "f");
  unsigned i_g = hg.add_function<Sum>({i_f, i_b}, "g");
  unsigned i_h = hg.add_function<Tanh>({i_g}, "h");
  unsigned i_p = hg.add_function<MatrixMultiply>({i_V, i_h}, "p");
  unsigned i_y_pred = hg.add_function<Sum>({i_p, i_a}, "y_pred");
  hg.add_function<SquaredEuclideanDistance>({i_y_pred, i_y}, "err");
  hg.PrintGraphviz();

  // train the parameters
  for (unsigned iter = 0; iter < 4; ++iter) {
    p_x(0,0) = (iter % 2 == 0) ? -1 : 1;
    p_x(1,0) = ((iter / 2) % 2 == 0) ? -1 : 1;
    double y = -1;
    if (iter % 2 != (iter / 2) % 2) y = -1;
    p_y(0,0) = y;
    cerr << "E = " << hg.forward() << endl;
  }
}

