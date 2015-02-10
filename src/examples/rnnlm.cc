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

unsigned DIM = 150;
unsigned VOCAB_SIZE = 30;

int main(int argc, char** argv) {
  sranddev();
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " text.ints\n";
    return 1;
  }
  vector<vector<unsigned>> corpus;
  string line;
  ifstream in(argv[1]);
  while(getline(in, line)) {
    istringstream is(line);
    vector<unsigned> x;
    unsigned v;
    while (is >> v) {
      assert(v < VOCAB_SIZE);
      x.push_back(v);
    }
    corpus.push_back(x);
  }

  Trainer* sgd = 0;
  if (false) {
    sgd = new RMSPropTrainer;
  } else {
    sgd = new SimpleSGDTrainer;
  }

  // parameters
  LookupParameters p_c(VOCAB_SIZE, Dim(DIM, 1));
  Parameters p_R(Dim(VOCAB_SIZE, DIM));
  Parameters p_bias(Dim(VOCAB_SIZE, 1));
  sgd->add_params(&p_c);
  sgd->add_params({&p_R, &p_bias});
  RNNBuilder rnn(2, DIM, DIM, sgd);

  for (unsigned iter = 0; iter < 1000; ++iter) {
    Timer iteration("epoch completed in");
    double loss = 0;
    unsigned lines = 0;
    unsigned chars = 0;
    for (auto& sent : corpus) {
      Hypergraph hg;
      rnn.add_parameter_edges(&hg);
      unsigned i_R = hg.add_parameter(&p_R, "R");
      unsigned i_bias = hg.add_parameter(&p_bias, "bias");
      vector<unsigned> errs;
      const unsigned slen = sent.size() - 1;
      for (unsigned t = 0; t < slen; ++t) {
        string ts = to_string(t);
        unsigned* wt;
        unsigned i_rwt = hg.add_lookup(&p_c, &wt, "x_" + ts);
        *wt = sent[t]; // input
        unsigned i_yt = rnn.add_input(i_rwt, &hg);
#if 1
        unsigned i_r1 = hg.add_function<MatrixMultiply>({i_R, i_yt}, "r1t_" + ts);
        unsigned i_r = hg.add_function<Sum>({i_r1, i_bias}, "rt_" + ts);
#else
        unsigned i_r = hg.add_function<Multilinear>({i_bias, i_R, i_yt}, "rt_" + ts);
#endif
        unsigned i_ydist = hg.add_function<LogSoftmax>({i_r}, "ydist_" + ts);  
        ConstParameters* p_ytrue_t = new ConstParameters(sent[t+1]);  // predict sent[t+1]
        unsigned i_ytrue = hg.add_input(p_ytrue_t, "ytrue_" + ts);
        errs.push_back(hg.add_function<PickElement>({i_ydist, i_ytrue}, "nerr_" + ts));
        chars++;
      }
      unsigned i_nerr = hg.add_function<Sum>(errs, "nerr");
      hg.add_function<Negate>({i_nerr}, "err");
      loss += hg.forward()(0,0);
      hg.backward();
      sgd->update(1.0 / slen);
      ++lines;
      if (lines == 10000) break;
    }
    cerr << "E = " << (loss / chars);
    
  }
}

