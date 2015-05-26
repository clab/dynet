#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  unsigned CONTEXT = 3;
  unsigned DIM = 100;
  unsigned VOCAB_SIZE = 29;

  // parameters
  Model model;
  SimpleSGDTrainer sgd(&model);
  LookupParameters* p_c = model.add_lookup_parameters(VOCAB_SIZE, {DIM});

  ComputationGraph cg;

  unsigned in_c1, in_c2, in_c3;  // set these to set the context words
  Expression c1 = lookup(cg, p_c, &in_c1);
  Expression c2 = lookup(cg, p_c, &in_c2);
  Expression c3 = lookup(cg, p_c, &in_c3);
  Expression C1 = parameter(cg, model.add_parameters({DIM, DIM}));
  Expression C2 = parameter(cg, model.add_parameters({DIM, DIM}));
  Expression C3 = parameter(cg, model.add_parameters({DIM, DIM}));
  Expression hb = parameter(cg, model.add_parameters({DIM}));
  Expression R = parameter(cg, model.add_parameters({VOCAB_SIZE, DIM}));
  unsigned ytrue;  // set ytrue to change the value of the input
  Expression bias = parameter(cg, model.add_parameters({VOCAB_SIZE}));

  Expression r = hb + C1 * c1 + C2 * c2 + C3 * c3;
  Expression nl = rectify(r);
  Expression o2 = bias + R * nl;
  Expression ydist = log_softmax(o2);
  Expression nerr = -pick(ydist, ytrue);
  cg.PrintGraphviz();

  // load some training data
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " ngrams.txt\n";
    return 1;
  }
  ifstream in(argv[1]);
  vector<vector<unsigned>> corpus;
  string line;
  while(getline(in, line)) {
    istringstream is(line);
    vector<unsigned> x(CONTEXT+1);
    for (unsigned i = 0; i <= CONTEXT; ++i) {
      is >> x[i];
      assert(x[i] < VOCAB_SIZE);
    }
    corpus.push_back(x);
  }


  // train the parameters
  for (unsigned iter = 0; iter < 100; ++iter) {
    Timer iteration("epoch completed in");
    double loss = 0;
    unsigned n = 0;
    for (auto& ci : corpus) {
      in_c1 = ci[0];
      in_c2 = ci[1];
      in_c3 = ci[2];
      ytrue  = ci[3];
      loss += as_scalar(cg.forward());
      cg.backward();
      ++n;
      sgd.update(1.0);
      if (n == 2500) break;
    }
    loss /= n;
    cerr << "E = " << loss << ' ';
  }
}

