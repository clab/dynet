#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"

#include <iostream>
#include <chrono>
#include <cassert>

using namespace std;
using namespace std::chrono;
using namespace dynet;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  size_t SEQ_LENGTH =40;
  size_t BATCH_SIZE=50;
  unsigned int HIDDEN=200;
  unsigned int NVOCAB=1000;
  unsigned int NCLASSES=300;
  unsigned int EMBED_SIZE=200;
  size_t N_SEQS=1000;
  assert(N_SEQS % BATCH_SIZE == 0);
  
  auto random_seq = [](size_t ln, size_t t) {
    vector<unsigned> ret(ln);
    uniform_int_distribution<unsigned> dist(0, t-1);
    for(size_t i = 0; i < ln; ++i) ret[i] = dist(*rndeng);
    return ret;
  };
  vector<vector<unsigned> > Xs(N_SEQS), Ys(N_SEQS);
  for(size_t i = 0; i < N_SEQS; ++i) {
    Xs[i] = random_seq(SEQ_LENGTH, NVOCAB);
    Ys[i] = random_seq(SEQ_LENGTH, NCLASSES);
  }
  
  ParameterCollection m;
  SimpleSGDTrainer trainer(m);
  
  LookupParameter E = m.add_lookup_parameters(NVOCAB, {EMBED_SIZE});
  VanillaLSTMBuilder fwR = VanillaLSTMBuilder(1, EMBED_SIZE, HIDDEN, m);
  VanillaLSTMBuilder bwR = VanillaLSTMBuilder(1, EMBED_SIZE, HIDDEN, m);
  Parameter T_= m.add_parameters({HIDDEN, HIDDEN*2});
  VanillaLSTMBuilder fwR2 = VanillaLSTMBuilder(1, HIDDEN, HIDDEN, m);
  VanillaLSTMBuilder bwR2 = VanillaLSTMBuilder(1, HIDDEN, HIDDEN, m);
  Parameter W_= m.add_parameters({NCLASSES, HIDDEN*2});
  
  // Must copy and paste this code because lambdas can't be templated in C++11
  auto transduce = [&](ComputationGraph & cg, vector<unsigned> & seq, vector<unsigned> & Y, Expression & T, Expression & W) {
    vector<Expression> seqE(seq.size()), fw(seq.size()), bw(seq.size()), zs(seq.size()), losses(seq.size());
    for(size_t i = 0; i < seq.size(); ++i) seqE[i] = lookup(cg, E, seq[i]);
    fwR.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) fw[i] = fwR.add_input(seqE[i]);
    bwR.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) bw[i] = bwR.add_input(seqE[seq.size()-i-1]);
    for(size_t i = 0; i < seq.size(); ++i) zs[i] = T * concatenate({fw[i], bw[seq.size()-i-1]});
    fwR2.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) fw[i] = fwR2.add_input(zs[i]);
    bwR2.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) bw[i] = bwR2.add_input(zs[seq.size()-i-1]);
    for(size_t i = 0; i < seq.size(); ++i) zs[i] = W * concatenate({fw[i], bw[seq.size()-i-1]});
    for(size_t i = 0; i < seq.size(); ++i) losses[i] = pickneglogsoftmax(zs[i], Y[i]);
    return sum(losses);
  };
  auto transduce_batch = [&](ComputationGraph & cg, vector<vector<unsigned>> & seq, vector<vector<unsigned>> & Y, Expression & T, Expression & W) {
    vector<Expression> seqE(seq.size()), fw(seq.size()), bw(seq.size()), zs(seq.size()), losses(seq.size());
    for(size_t i = 0; i < seq.size(); ++i) seqE[i] = lookup(cg, E, seq[i]);
    fwR.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) fw[i] = fwR.add_input(seqE[i]);
    bwR.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) bw[i] = bwR.add_input(seqE[seq.size()-i-1]);
    for(size_t i = 0; i < seq.size(); ++i) zs[i] = T * concatenate({fw[i], bw[seq.size()-i-1]});
    fwR2.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) fw[i] = fwR2.add_input(zs[i]);
    bwR2.start_new_sequence();
    for(size_t i = 0; i < seq.size(); ++i) bw[i] = bwR2.add_input(zs[seq.size()-i-1]);
    for(size_t i = 0; i < seq.size(); ++i) zs[i] = W * concatenate({fw[i], bw[seq.size()-i-1]});
    for(size_t i = 0; i < seq.size(); ++i) losses[i] = pickneglogsoftmax(zs[i], Y[i]);
    return sum(losses);
  };

  string yes = "batch";
  bool man_batch = (argc >= 2 && yes == argv[1]);
  int last_step = (argc >= 3 ? atoi(argv[2]) : 2);

  vector<vector<unsigned>> bXs(SEQ_LENGTH, vector<unsigned>(BATCH_SIZE));
  vector<vector<unsigned>> bYs(SEQ_LENGTH, vector<unsigned>(BATCH_SIZE));
  vector<Expression> batch(BATCH_SIZE);
  time_point<system_clock> start = system_clock::now();
  for(size_t i = 0; i < Xs.size(); ) {
    Expression s;
    ComputationGraph cg;
    Expression T = parameter(cg, T_), W = parameter(cg, W_);
    fwR.new_graph(cg); bwR.new_graph(cg);
    fwR2.new_graph(cg); bwR2.new_graph(cg);
    // Do manual batching
    if(man_batch) {
      for(size_t b = 0; b < BATCH_SIZE; ++i, ++b) {
        for(size_t s = 0; s < SEQ_LENGTH; ++s) {
          bXs[s][b] = Xs[b][s];
          bYs[s][b] = Ys[b][s];
        }
      }
      s = sum_batches(transduce_batch(cg, bXs, bYs, T, W));
    // Do not manual batching
    } else {
      do {
        batch[i % BATCH_SIZE] = transduce(cg, Xs[i], Ys[i], T, W);
      } while (++i % BATCH_SIZE != 0);
      s = sum(batch);
    }
    cg.forward(s);
    if(last_step > 0) {
      cg.backward(s);
      if(last_step > 1)
        trainer.update();
    }
  }
  std::chrono::duration<float> fs = (system_clock::now() - start);
  cerr << "sent/sec: " << Xs.size() / (duration_cast<milliseconds>(fs).count()/float(1000)) << endl;
  cerr << "sec/sent: " << (duration_cast<milliseconds>(fs).count()/float(1000)) / Xs.size() << endl;

}
