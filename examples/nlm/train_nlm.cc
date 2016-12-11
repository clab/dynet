#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  unsigned CONTEXT = 3;
  unsigned DIM = 100;
  unsigned VOCAB_SIZE = 29;

  // parameters
  Model model;
  SimpleSGDTrainer sgd(model);
  LookupParameter p_c = model.add_lookup_parameters(VOCAB_SIZE, {DIM});

  ComputationGraph cg;

  vector<unsigned> in_c(CONTEXT); // set these to set the context words
  vector<Expression> c(CONTEXT);
  for (unsigned i=0; i<CONTEXT; ++i)
    c[i] = lookup(cg, p_c, &in_c[i]);

  Expression C = parameter(cg, model.add_parameters({DIM, DIM*CONTEXT}));
  Expression hb = parameter(cg, model.add_parameters({DIM}));
  Expression R = parameter(cg, model.add_parameters({VOCAB_SIZE, DIM}));
  unsigned ytrue;  // set ytrue to change the value of the input
  Expression bias = parameter(cg, model.add_parameters({VOCAB_SIZE}));

  Expression cc = concatenate(c);
  Expression r = hb + C * cc;
  Expression nl = rectify(r);
  Expression o2 = bias + R * nl;
  Expression ydist = log_softmax(o2);
  Expression nerr = -pick(ydist, &ytrue);
  cg.print_graphviz();

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
      copy(ci.begin(), ci.begin()+CONTEXT, in_c.begin());
      ytrue  = ci.back();
      loss += as_scalar(cg.forward(nerr));
      cg.backward(nerr);
      ++n;
      sgd.update(1.0);
      if (n == 2500) break;
    }
    loss /= n;
    cerr << "E = " << loss << ' ';
  }
}

