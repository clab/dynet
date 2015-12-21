#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/cfsm-builder.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 256;  //256
unsigned HIDDEN_DIM = 256;  // 1024
unsigned VOCAB_SIZE = 0;

cnn::Dict d;
int kSOS;
int kEOS;

template <class Builder>
struct RNNLanguageModel {
  LookupParameters* p_c;
  Builder builder;
  ClassFactoredSoftmaxBuilder& cfsm;
  explicit RNNLanguageModel(Model& model, ClassFactoredSoftmaxBuilder& h) :
      p_c(model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
      cfsm(h) {}

  // return Expression of total loss
  Expression BuildLMGraph(const vector<int>& sent, ComputationGraph& cg) {
    const unsigned slen = sent.size() - 1;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    cfsm.new_graph(cg);
    //Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    //Expression i_bias = parameter(cg, p_bias);  // word bias
    vector<Expression> errs;
    for (unsigned t = 0; t < slen; ++t) {
      Expression i_x_t = lookup(cg, p_c, sent[t]);
      // y_t = RNN(x_t)
      Expression i_y_t = builder.add_input(i_x_t);
      Expression i_err = cfsm.neg_log_softmax(i_y_t, sent[t+1]);
      errs.push_back(i_err);
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  }

  // return Expression for total loss
  void RandomSample(int max_len = 150) {
    cerr << endl;
    ComputationGraph cg;
    cfsm.new_graph(cg);
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    vector<Expression> errs;
    int len = 0;
    int cur = kSOS;
    while(len < max_len && cur != kEOS) {
      ++len;
      Expression i_x_t = lookup(cg, p_c, cur);
      // y_t = RNN(x_t)
      Expression i_y_t = builder.add_input(i_x_t);
      cur = cfsm.sample(i_y_t);
      cerr << (len == 1 ? "" : " ") << d.Convert(cur);
    }
    cerr << endl;
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 4 && argc != 5) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt clusters.txt [model.params]\n";
    return 1;
  }
  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  Model model;
  ClassFactoredSoftmaxBuilder cfsm(HIDDEN_DIM, argv[3], &d, &model);
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
    ifstream in(argv[2]);
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
  ostringstream os;
  os << "cfsmlm"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  bool use_momentum = false;
  Trainer* sgd = nullptr;
  //if (use_momentum)
  //  sgd = new MomentumSGDTrainer(&model);
  //else
  sgd = new SimpleSGDTrainer(&model);

  RNNLanguageModel<LSTMBuilder> lm(model, cfsm);
  //RNNLanguageModel<SimpleRNNBuilder> lm(model, cfsm);
  bool has_loaded_model = false;
  if (argc == 5) {
    has_loaded_model = true;
    string fname = argv[4];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 10;
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
#if 1
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }

      // build graph for this instance
      ComputationGraph cg;
      auto& sent = training[order[si]];
      chars += sent.size() - 2;
      ++si;
      lm.BuildLMGraph(sent, cg);
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update();
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
    lm.RandomSample();

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
#endif
      double dloss = 0;
      int dchars = 0;
      for (auto& sent : dev) {
        ComputationGraph cg;
        lm.BuildLMGraph(sent, cg);
        dloss += as_scalar(cg.forward());
        dchars += sent.size() - 2;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
#if 1
      if (dloss < best) {
        best = dloss;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
    }
#endif
  }
  delete sgd;
}

