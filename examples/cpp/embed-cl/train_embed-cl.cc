#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/access.hpp>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

unsigned REP_DIM = 128;
unsigned INPUT_VOCAB_SIZE = 0;
unsigned OUTPUT_VOCAB_SIZE = 0;

dynet::Dict sd;
dynet::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTRG_SOS;
int kTRG_EOS;

struct Encoder {
  LookupParameter p_s;
  LookupParameter p_t;
  vector<Expression> m;

  Encoder() {}

  explicit Encoder(Model& model) {
    p_s = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {REP_DIM}); 
    p_t = model.add_lookup_parameters(OUTPUT_VOCAB_SIZE, {REP_DIM}); 
  }

  Expression EmbedSource(const vector<int>& sent, ComputationGraph& cg) {
    m.resize(sent.size() + 2);
    m[0] = lookup(cg, p_s, kSRC_SOS);
    int i = 1;
    for (auto& w : sent)
      m[i++] = lookup(cg, p_s, w);
    m[i] = lookup(cg, p_s, kSRC_EOS);
#define DUMB_ADDITIVE
#ifdef DUMB_ADDITIVE
    return sum(m);
#else
    return sum_cols(tanh(kmh_ngram(concatenate_cols(m), 2)));
#endif
  }

  Expression EmbedTarget(const vector<int>& sent, ComputationGraph& cg) {
    m.resize(sent.size() + 2);
    m[0] = lookup(cg, p_s, kTRG_SOS);
    int i = 1;
    for (auto& w : sent)
      m[i++] = lookup(cg, p_t, w);
    m[i] = lookup(cg, p_s, kTRG_EOS);
#ifdef DUMB_ADDITIVE
    return sum(m);
#else
    return sum_cols(tanh(kmh_ngram(concatenate_cols(m), 2)));
#endif
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & p_s;
    ar & p_t;
  }
};

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  vector<pair<vector<int>, vector<int>>> training, dev;
  string line;
  kSRC_SOS = sd.convert("<s>");
  kSRC_EOS = sd.convert("</s>");
  kTRG_SOS = td.convert("<s>");
  kTRG_EOS = td.convert("</s>");
  int tlc = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      vector<int> src, trg;
      read_sentence_pair(line, src, sd, trg, td);
      training.push_back(make_pair(src, trg));
    }
    cerr << tlc << " lines, " << sd.size() << " source types, " << td.size() << " target types\n";
  }
  sd.freeze(); // no new word types allowed
  td.freeze(); // no new word types allowed
  INPUT_VOCAB_SIZE = sd.size();
  OUTPUT_VOCAB_SIZE = td.size();
#if 0

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      dev.push_back(read_sentence(line, &d));
      dtoks += dev.back().size();
      if (dev.back().front() != kSOS && dev.back().back() != kEOS) {
        cerr << "Dev sentence in " << argv[2] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  ostringstream os;
  os << "bilm"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;
#endif
  Model model;
  Encoder emb;
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model >> emb;
  }
  else {
    emb = Encoder(model);
  }

  bool use_momentum = false;
  Trainer* sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(model);
  else
    sgd = new SimpleSGDTrainer(model);

  unsigned report_every_i = 100;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  // int report = 0;
  // unsigned dev_every_i_reports = 10;
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
      ComputationGraph cg;
      auto& sent_pair = training[order[si]];
      ++si;
      Expression s = emb.EmbedSource(sent_pair.first, cg);
      Expression sim = squared_distance(s, emb.EmbedTarget(sent_pair.second, cg));
      float margin = 2;
      const unsigned K = 20;
      vector<Expression> noise(K);
      for (unsigned j = 0; j < K; ++j) {
        unsigned sample = rand01() * training.size();
        while (sample == order[si] || sample == training.size()) { sample = rand01() * training.size(); }
        Expression sim_n = squared_distance(s, emb.EmbedTarget(training[sample].second, cg));
        noise[j] = pairwise_rank_loss(sim, sim_n, margin);
      }
      Expression l = sum(noise);
      auto iloss = as_scalar(cg.forward(l));
      assert(iloss >= 0);
      if (iloss > 0) {
        loss += iloss;
        cg.backward(l);
        sgd->update();
      }
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss) << " ppl=" << exp(loss / chars) << ' ';

#if 0
    lm.RandomSample();
#endif
#if 0
    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dchars = 0;
      for (auto& sent : dev) {
        ComputationGraph cg;
        lm.BuildGraph(sent, sent, cg);
        dloss += as_scalar(cg.forward());
        dchars += sent.size() - 1;
      }
      if (dloss < best) {
        best = dloss;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model << emb;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
    }
#endif
  }
  delete sgd;
}

