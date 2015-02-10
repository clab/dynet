#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"

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

  unsigned DIM = 100;
  unsigned VOCAB_SIZE = 29;

  // parameters
  LookupParameters p_c(VOCAB_SIZE, Dim(DIM, 1));
  Parameters p_R(Dim(VOCAB_SIZE, DIM));
  Parameters p_bias(Dim(VOCAB_SIZE, 1));
  sgd->add_params(&p_c);
  sgd->add_params({&p_R, &p_bias});

  { Timer ttt("graph constructed in ");
  Hypergraph hg;
  unsigned i_R = hg.add_parameter(&p_R, "R");
  unsigned i_bias = hg.add_parameter(&p_bias, "bias");
  //RNNBuilder rnn(&hg, 1, DIM, DIM, sgd);
  LSTMBuilder rnn(&hg, 1, DIM, DIM, sgd);
  vector<unsigned> errs;
  for (unsigned t = 0; t < 5; ++t) {
    string ts = to_string(t);
    unsigned* wt;
    unsigned i_rwt = hg.add_lookup(&p_c, &wt, "x_" + ts);
    unsigned i_yt = rnn.add_input(i_rwt);
#if 0
    unsigned i_r1 = hg.add_function<MatrixMultiply>({i_R, i_yt}, "r1t_" + ts);
    unsigned i_r = hg.add_function<Sum>({i_r1, i_bias}, "rt_" + ts);
#else
    unsigned i_r = hg.add_function<Multilinear>({i_bias, i_R, i_yt}, "rt_" + ts);
#endif
    unsigned i_ydist = hg.add_function<LogSoftmax>({i_r}, "ydist_" + ts);  
    ConstParameters* p_ytrue_t = new ConstParameters(Dim(1,1));
    unsigned i_ytrue = hg.add_input(p_ytrue_t, "ytrue_" + ts);
    errs.push_back(hg.add_function<PickElement>({i_ydist, i_ytrue}, "nerr_" + ts));
  }
  unsigned i_nerr = hg.add_function<Sum>(errs, "nerr");
  hg.add_function<Negate>({i_nerr}, "err");
  hg.PrintGraphviz();
  }


#if 0
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
#endif
}

