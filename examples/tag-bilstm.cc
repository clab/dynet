#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
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

float pdrop = 0.5;
unsigned LAYERS = 1;
unsigned INPUT_DIM = 128;
unsigned HIDDEN_DIM = 128;
unsigned TAG_HIDDEN_DIM = 32;
unsigned TAG_DIM = 32;
unsigned TAG_SIZE = 0;
unsigned VOCAB_SIZE = 0;

bool eval = false;
cnn::Dict d;
cnn::Dict td;
int kNONE;
int kSOS;
int kEOS;

template <class Builder>
struct RNNLanguageModel {
  LookupParameters* p_w;
  Parameters* p_l2th;
  Parameters* p_r2th;
  Parameters* p_thbias;

  Parameters* p_th2t;
  Parameters* p_tbias;
  Builder l2rbuilder;
  Builder r2lbuilder;
  explicit RNNLanguageModel(Model& model) :
      l2rbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
      r2lbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
    p_w = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
    p_l2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
    p_r2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
    p_thbias = model.add_parameters({TAG_HIDDEN_DIM});

    p_th2t = model.add_parameters({TAG_SIZE, TAG_HIDDEN_DIM});
    p_tbias = model.add_parameters({TAG_SIZE});
  }

  // return Expression of total loss
  Expression BuildTaggingGraph(const vector<int>& sent, const vector<int>& tags, ComputationGraph& cg, double* cor = 0, unsigned* ntagged = 0) {
    const unsigned slen = sent.size();
    l2rbuilder.new_graph(cg);  // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);  // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();
    Expression i_l2th = parameter(cg, p_l2th);
    Expression i_r2th = parameter(cg, p_r2th);
    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_th2t = parameter(cg, p_th2t);
    Expression i_tbias = parameter(cg, p_tbias); 
    vector<Expression> errs;
    vector<Expression> i_words(slen);
    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);

    // read sequence from left to right
    l2rbuilder.add_input(lookup(cg, p_w, kSOS));
    for (unsigned t = 0; t < slen; ++t) {
      i_words[t] = lookup(cg, p_w, sent[t]);
      if (!eval) { i_words[t] = noise(i_words[t], 0.1); }
      fwds[t] = l2rbuilder.add_input(i_words[t]);
    }

    // read sequence from right to left
    r2lbuilder.add_input(lookup(cg, p_w, kEOS));
    for (unsigned t = 0; t < slen; ++t)
      revs[slen - t - 1] = r2lbuilder.add_input(i_words[slen - t - 1]);

    for (unsigned t = 0; t < slen; ++t) {
      if (tags[t] != kNONE) {
        if (ntagged) (*ntagged)++;
        Expression i_th = tanh(affine_transform({i_thbias, i_l2th, fwds[t], i_r2th, revs[t]}));
        //if (!eval) { i_th = dropout(i_th, pdrop); }
        Expression i_t = affine_transform({i_tbias, i_th2t, i_th});
        if (cor) {
          vector<float> dist = as_vector(cg.incremental_forward());
          double best = -9e99;
          int besti = -1;
          for (int i = 0; i < dist.size(); ++i) {
            if (dist[i] > best) { best = dist[i]; besti = i; }
          }
          if (tags[t] == besti) (*cor)++;
        }
        if (tags[t] != kNONE) {
          Expression i_err = pickneglogsoftmax(i_t, tags[t]);
          errs.push_back(i_err);
        }
      }
    }
    return sum(errs);
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  kNONE = td.Convert("*");
  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  vector<pair<vector<int>,vector<int>>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      int nc = 0;
      vector<int> x,y;
      ReadSentencePair(line, &x, &d, &y, &td);
      assert(x.size() == y.size());
      if (x.size() == 0) { cerr << line << endl; abort(); }
      training.push_back(make_pair(x,y));
      for (unsigned i = 0; i < y.size(); ++i) {
        if (y[i] != kNONE) { ++nc; }
      }
      if (nc == 0) {
        cerr << "No tagged tokens in line " << tlc << endl;
        abort();
      }
      ttoks += x.size();
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
    cerr << "Tags: " << td.size() << endl;
  }
  d.Freeze(); // no new word types allowed
  td.Freeze(); // no new tag types allowed
  VOCAB_SIZE = d.size();
  TAG_SIZE = td.size();

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      vector<int> x,y;
      ReadSentencePair(line, &x, &d, &y, &td);
      assert(x.size() == y.size());
      dev.push_back(make_pair(x,y));
      dtoks += x.size();
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  ostringstream os;
  os << "tagger"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  Model model;
  bool use_momentum = true;
  Trainer* sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(&model);
  else
    sgd = new SimpleSGDTrainer(&model);

  RNNLanguageModel<LSTMBuilder> lm(model);
  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

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
    unsigned ttags = 0;
    double correct = 0;
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
      lm.BuildTaggingGraph(sent.first, sent.second, cg, &correct, &ttags);
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update(1.0);
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / ttags) << ") ";

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dtags = 0;
      double dcorr = 0;
      eval = true;
      //lm.p_th2t->scale_parameters(pdrop);
      for (auto& sent : dev) {
        ComputationGraph cg;
        lm.BuildTaggingGraph(sent.first, sent.second, cg, &dcorr, &dtags);
        dloss += as_scalar(cg.forward());
      }
      //lm.p_th2t->scale_parameters(1/pdrop);
      eval = false;
      if (dloss < best) {
        best = dloss;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << " acc=" << (dcorr / dtags) << ' ';
    }
  }
  delete sgd;
}

