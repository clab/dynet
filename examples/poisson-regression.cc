#include "cnn/grad-check.h"
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/deep-lstm.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
# include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 16;  //256
unsigned HIDDEN_DIM = 32;  // 1024
unsigned VOCAB_SIZE = 0;

cnn::Dict d;
int kSOS;
int kEOS;

template <class Builder>
struct RNNLengthPredictor {
  LookupParameters* p_c;
  Parameters* p_R;
  Parameters* p_bias;
  Builder builder;
  explicit RNNLengthPredictor(Model& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
    p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
    p_R = model.add_parameters({1, HIDDEN_DIM});
    p_bias = model.add_parameters({1});
  }

  // return Expression of total loss
  Expression BuildLMGraph(const vector<int>& sent, unsigned len, ComputationGraph& cg, bool flag = false) {
    const unsigned slen = sent.size() - 1;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    Expression R = parameter(cg, p_R);
    Expression bias = parameter(cg, p_bias);
    vector<Expression> errs;
    for (unsigned t = 0; t < slen; ++t) {
      Expression i_x_t = lookup(cg, p_c, sent[t]);
      // y_t = RNN(x_t)
      builder.add_input(i_x_t);
    }
    Expression pred = affine_transform({bias, R, builder.back()});
    if (flag) {
      unsigned x = exp(as_scalar(cg.incremental_forward()));
      cerr << "PRED=" << x << " TRUE=" << len << "\t(DIFF=" << ((int)x - (int)len) << ")" << endl;
    }
    return poisson_loss(pred, len);
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  vector<pair<vector<int>,unsigned>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    Dict td;
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      vector<int> x, ty;
      ReadSentencePair(line, &x, &d, &ty, &td);
      assert(ty.size() == 1);
      const string& v = td.Convert(ty[0]);
      for(auto c : v) { assert(c >= '0' && c <= '9'); }
      unsigned y = atoi(v.c_str());
      training.push_back(make_pair(x,y));
      ttoks += training.back().first.size();
      if (training.back().first.front() != kSOS && training.back().first.back() != kEOS) {
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
    Dict td;
    while(getline(in, line)) {
      ++dlc;
      vector<int> x, ty;
      ReadSentencePair(line, &x, &d, &ty, &td);
      assert(ty.size() == 1);
      const string& v = td.Convert(ty[0]);
      for(auto c : v) { assert(c >= '0' && c <= '9'); }
      unsigned y = atoi(v.c_str());
      dev.push_back(make_pair(x,y));
      dtoks += dev.back().first.size();
      if (dev.back().first.front() != kSOS && dev.back().first.back() != kEOS) {
        cerr << "Dev sentence in " << argv[2] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  ostringstream os;
  os << "lm"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  Model model;
  Trainer* sgd = nullptr;
  sgd = new SimpleSGDTrainer(&model);

  RNNLengthPredictor<LSTMBuilder> lm(model);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 20;
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
        shuffle(order.begin(), order.end(), *rndeng);
      }

      // build graph for this instance
      ComputationGraph cg;
      auto& sent = training[order[si]];
      ++si;
      lm.BuildLMGraph(sent.first, sent.second, cg);
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update();
      ++lines;
      ++chars;
    }
    sgd->status();
    cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dchars = 0;
      for (auto& sent : dev) {
        ComputationGraph cg;
        lm.BuildLMGraph(sent.first, sent.second, cg, true);
        dloss += as_scalar(cg.forward());
        dchars++;
      }
      if (dloss < best) {
        best = dloss;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
    }
  }
  delete sgd;
}

