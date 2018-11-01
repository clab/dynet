#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/globals.h"
#include "dynet/io.h"
#include "getpid.h"

using namespace std;
using namespace dynet;

float pdrop = 0.5;
unsigned LAYERS = 1;
unsigned INPUT_DIM = 128;
unsigned HIDDEN_DIM = 128;
unsigned TAG_HIDDEN_DIM = 32;
unsigned TAG_DIM = 32;
unsigned TAG_SIZE = 0;
unsigned VOCAB_SIZE = 0;

bool use_momentum = false;
bool use_ema = false;
bool use_cma = false;

bool eval = false;
dynet::Dict d;
dynet::Dict td;
int kNONE;
int kSOS;
int kEOS;

template <class Builder>
struct RNNLanguageModel {
  LookupParameter p_w;
  Parameter p_l2th;
  Parameter p_r2th;
  Parameter p_thbias;

  Parameter p_th2t;
  Parameter p_tbias;
  Builder l2rbuilder;
  Builder r2lbuilder;
  explicit RNNLanguageModel(ParameterCollection& model) :
      l2rbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model),
      r2lbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model) {
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

    // set words, adding noise during training (non-eval)
    for (unsigned t = 0; t < slen; ++t) {
      i_words[t] = lookup(cg, p_w, sent[t]);
      if (!eval) { i_words[t] = noise(i_words[t], 0.1); }
    }

    // read sequence from left to right
    l2rbuilder.add_input(lookup(cg, p_w, kSOS));
    for (unsigned t = 0; t < slen; ++t)
      fwds[t] = l2rbuilder.add_input(i_words[t]);
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
          vector<float> dist = as_vector(cg.incremental_forward(i_t));

          // Find best tag according to the distribution
          double best = -9e99;
          int besti = -1;
          for (int i = 0; i < static_cast<int>(dist.size()); ++i) {
            if (dist[i] > best) { best = dist[i]; besti = i; }
          }
          if (tags[t] == besti) (*cor)++;
        }

        Expression i_err = pickneglogsoftmax(i_t, tags[t]);
        errs.push_back(i_err);
      }
    }
    return sum(errs);
  }
};

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  std::string path_corpus;
  std::string path_dev;
  std::string path_model;

  bool opt_succeed = true;
  unsigned n_pos_args = 0;
  std::string* pos_args[]{&path_corpus, &path_dev, &path_model};
  for (int argi = 1 ; argi < argc ; ++argi)
  {
    std::string opt(argv[argi]);
    if (opt[0] != '-')
    {
        if (n_pos_args == 3)
        {
           opt_succeed = false;
           break;
        }
        *(pos_args[n_pos_args]) = opt;
        ++n_pos_args;
    }
    else if (opt == "--momentum")
    {
        use_momentum = true;
    }
    else if (opt == "--ema")
    {
        use_ema = true;
    }
    else if (opt == "--cma")
    {
        use_cma = true;
    }
    else
    {
        opt_succeed = false;
        break;
    }
  }

  if (use_ema && use_cma)
  {
    cerr << "Can not use both Exponential Moving Average and Cumulative Moving Average\n";
    opt_succeed = false;
  }
  if (path_corpus.size() == 0 || path_dev.size() == 0)
    opt_succeed = false;

  if (!opt_succeed)
  {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params] [--momentum] [--ema] [--cma]\n";
    return 1;
  }
  cerr << "Training data: " << path_corpus << "\n";
  cerr << "Dev data: " << path_dev << "\n";
  if (path_model.size() != 0)
    cerr << "Model params: " << path_corpus << "\n";
  cerr << "Optimizer: SGD";
  if (use_momentum) cerr << "+momentum";
  if (use_ema) cerr << "+ema";
  if (use_cma) cerr << "+cma";
  cerr << "\n";

  kNONE = td.convert("*");
  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");
  vector<pair<vector<int>,vector<int>>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << path_corpus << "...\n";
  {
    ifstream in(path_corpus);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      int nc = 0;
      vector<int> x,y;
      read_sentence_pair(line, x, d, y, td);
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
  d.freeze(); // no new word types allowed
  td.freeze(); // no new tag types allowed
  d.set_unk("UNKNOWN_WORD");
  VOCAB_SIZE = d.size();
  TAG_SIZE = td.size();

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << path_dev<< "...\n";
  {
    ifstream in(path_dev);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      vector<int> x,y;
      read_sentence_pair(line, x, d, y, td);
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
     << (use_momentum ? "-momentum" : "")
     << (use_ema ? "-ema" : "")
     << "-pid" << getpid()
     << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  ParameterCollection model;
  std::unique_ptr<Trainer> trainer;
  if (use_momentum)
    trainer.reset(new MomentumSGDTrainer(model));
  else
    trainer.reset(new SimpleSGDTrainer(model));
  if (use_ema)
    trainer->exponential_moving_average(0.999);
  if (use_cma)
    trainer->cumulative_moving_average();

  RNNLanguageModel<LSTMBuilder> lm(model);
  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  if (path_model.size() != 0) {
    TextFileLoader loader(path_model);
    loader.populate(model);
  }

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 25;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
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
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }

      // build graph for this instance
      ComputationGraph cg;
      auto& sent = training[order[si]];
      ++si;
      Expression loss_expr = lm.BuildTaggingGraph(sent.first, sent.second, cg, &correct, &ttags);

      // Run forward pass, backpropagate, and do an update
      loss += as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      trainer->update();
      ++lines;
    }
    trainer->status();
    cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / ttags) << ") ";

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      if (use_ema || use_cma)
        trainer->swap_params_to_moving_average(true, true);
      double dloss = 0;
      unsigned dtags = 0;
      double dcorr = 0;
      eval = true;
      //lm.p_th2t->scale_parameters(pdrop);
      for (auto& sent : dev) {
        ComputationGraph cg;
        Expression loss_expr = lm.BuildTaggingGraph(sent.first, sent.second, cg, &dcorr, &dtags);
        dloss += as_scalar(cg.forward(loss_expr));
      }
      //lm.p_th2t->scale_parameters(1/pdrop);
      eval = false;
      if (dloss < best) {
        best = dloss;
        TextFileSaver saver(fname);
        saver.save(model);
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << " acc=" << (dcorr / dtags) << ' ';
      if (use_ema)
        trainer->swap_params_to_weights();
    }
  }
}
