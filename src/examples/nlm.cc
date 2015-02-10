#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cnn;

int main(int argc, char** argv) {
  sranddev();

  Trainer* sgd = 0;
  if (false) {
    sgd = new RMSPropTrainer;
  } else {
    sgd = new SimpleSGDTrainer;
  }

  unsigned CONTEXT = 3;
  unsigned DIM = 100;
  unsigned VOCAB_SIZE = 29;

  // parameters
  LookupParameters p_c(VOCAB_SIZE, Dim(DIM, 1));
  Parameters p_C1(Dim(DIM, DIM));
  Parameters p_C2(Dim(DIM, DIM));
  Parameters p_C3(Dim(DIM, DIM));
  Parameters p_R(Dim(VOCAB_SIZE, DIM));
  Parameters p_bias(Dim(VOCAB_SIZE, 1));
  Parameters p_hb(Dim(DIM, 1));
  sgd->add_params(&p_c);
  sgd->add_params({&p_C1, &p_C2, &p_C3, &p_hb, &p_R, &p_bias});

  // inputs
  ConstParameters p_ytrue(Dim(1,1));

  // build the graph
  Hypergraph hg;
  unsigned *in_c1, *in_c2, *in_c3;
  unsigned i_c1 = hg.add_lookup(&p_c, &in_c1, "c1");
  unsigned i_c2 = hg.add_lookup(&p_c, &in_c2, "c2");
  unsigned i_c3 = hg.add_lookup(&p_c, &in_c3, "c3");
  unsigned i_C1 = hg.add_parameter(&p_C1, "C1");
  unsigned i_C2 = hg.add_parameter(&p_C2, "C2");
  unsigned i_C3 = hg.add_parameter(&p_C3, "C3");
  unsigned i_hb = hg.add_parameter(&p_hb, "hb");
  unsigned i_R = hg.add_parameter(&p_R, "R");
  unsigned i_ytrue = hg.add_input(&p_ytrue, "ytrue");
  unsigned i_bias = hg.add_parameter(&p_bias, "bias");
  unsigned i_r1 = hg.add_function<MatrixMultiply>({i_C1, i_c1}, "r1");
  unsigned i_r2 = hg.add_function<MatrixMultiply>({i_C2, i_c2}, "r2");
  unsigned i_r3 = hg.add_function<MatrixMultiply>({i_C3, i_c3}, "r3");
  unsigned i_r = hg.add_function<Sum>({i_r1, i_r2, i_r3, i_hb}, "r");
  unsigned i_nl = hg.add_function<Rectify>({i_r}, "nl");
  unsigned i_o1 = hg.add_function<MatrixMultiply>({i_R, i_nl}, "o1");
  unsigned i_o2 = hg.add_function<Sum>({i_o1, i_bias}, "o2");
  unsigned i_ydist = hg.add_function<LogSoftmax>({i_o2}, "ydist");
  unsigned i_nerr = hg.add_function<PickElement>({i_ydist, i_ytrue}, "nerr");
  hg.add_function<Negate>({i_nerr}, "err");
  hg.PrintGraphviz();

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
      *in_c1 = ci[0];
      *in_c2 = ci[1];
      *in_c3 = ci[2];
      p_ytrue(0,0) = ci[CONTEXT];
      loss += hg.forward()(0,0);
      hg.backward();
      ++n;
      sgd->update(1.0);
      if (n == 2500) break;
    }
    loss /= n;
    cerr << "E = " << loss << ' ';
  }
}

