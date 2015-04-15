#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm-fast.h"
#include "cnn/dict.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cnn;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 8;
unsigned HIDDEN_DIM = 24;
unsigned VOCAB_SIZE = 0;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt\n";
    return 1;
  }
  cnn::Dict d;
  const int kSOS = d.Convert("<s>");
  const int kEOS = d.Convert("</s>");
  vector<vector<int>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
    ++tlc;
    training.push_back(ReadSentence(line, &d));
    ttoks += training.back().size();
    if (training.back().front() != kSOS && training.back().back() != kEOS) {
      cerr << "Training sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
      abort();
    }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.Freeze(); // no new word types allowed
  VOCAB_SIZE = d.size();

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
    ++dlc;
    dev.push_back(ReadSentence(line, &d));
    dtoks += dev.back().size();
    if (dev.back().front() != kSOS && dev.back().back() != kEOS) {
      cerr << "Dev sentence in " << argv[2] << ":" << tlc << " didn't start or end with <s>, </s>\n";
      abort();
    }
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }

  Model model;
  bool use_momentum = false;
  Trainer* sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(&model);
  else
    sgd = new SimpleSGDTrainer(&model);

  // parameters
  LookupParameters* p_c = model.add_lookup_parameters(VOCAB_SIZE, Dim({INPUT_DIM})); 
  Parameters* p_R = model.add_parameters(Dim({VOCAB_SIZE, HIDDEN_DIM}));
  Parameters* p_bias = model.add_parameters(Dim({VOCAB_SIZE}));
  //RNNBuilder rnn(LAYERS, INPUT_DIM, HIDDEN_DIM, &model);
  LSTMBuilder_CIFG rnn(LAYERS, INPUT_DIM, HIDDEN_DIM, &model);

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 25;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned chars = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
      }

      // build graph for this instance
      Hypergraph hg;
      rnn.new_graph();  // reset RNN builder for new graph
      rnn.add_parameter_edges(&hg);  // add variables for its parameters
      rnn.start_new_sequence(&hg);
      VariableIndex i_R = hg.add_parameter(p_R); // hidden -> word rep parameter
      VariableIndex i_bias = hg.add_parameter(p_bias);  // word bias
      vector<VariableIndex> errs;
      auto& sent = training[order[si]];
      ++si;
      const unsigned slen = sent.size() - 1;
      for (unsigned t = 0; t < slen; ++t) {
        // x_t = lookup sent[t] in parameters p_c
        VariableIndex i_x_t = hg.add_lookup(p_c, sent[t]);
        // y_t = RNN(x_t)
        VariableIndex i_y_t = rnn.add_input(i_x_t, &hg);
        // r_t = bias + R * y_t
        VariableIndex i_r_t = hg.add_function<Multilinear>({i_bias, i_R, i_y_t});
        // ydist = softmax(r_t)
        VariableIndex i_ydist = hg.add_function<LogSoftmax>({i_r_t});
        errs.push_back(hg.add_function<PickElement>({i_ydist}, sent[t+1]));
        chars++;
      }
      VariableIndex i_nerr = hg.add_function<Sum>(errs);
      hg.add_function<Negate>({i_nerr});
      loss += as_scalar(hg.forward());
      hg.backward();
      sgd->update();
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dchars = 0;
      for (auto& sent : dev) {
        Hypergraph hg;
        rnn.new_graph();  // reset RNN builder for new graph
        rnn.add_parameter_edges(&hg);  // add variables for its parameters
        rnn.start_new_sequence(&hg);
        VariableIndex i_R = hg.add_parameter(p_R); // hidden -> word rep parameter
        VariableIndex i_bias = hg.add_parameter(p_bias);  // word bias
        vector<VariableIndex> errs;
        const unsigned slen = sent.size() - 1;
        for (unsigned t = 0; t < slen; ++t) {
          // x_t = lookup sent[t] in parameters p_c
          VariableIndex i_x_t = hg.add_lookup(p_c, sent[t]);
          // y_t = RNN(x_t)
          VariableIndex i_y_t = rnn.add_input(i_x_t, &hg);
          // r_t = bias + R * y_t
          VariableIndex i_r_t = hg.add_function<Multilinear>({i_bias, i_R, i_y_t});
          // ydist = softmax(r_t)
          VariableIndex i_ydist = hg.add_function<LogSoftmax>({i_r_t});
          errs.push_back(hg.add_function<PickElement>({i_ydist}, sent[t+1]));
          dchars++;
        }
        VariableIndex i_nerr = hg.add_function<Sum>(errs);
        hg.add_function<Negate>({i_nerr});
        dloss += as_scalar(hg.forward());
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
    }
  }
  delete sgd;
}

