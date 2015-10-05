#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 1;
unsigned CODE_DIM = 64;
unsigned CHAR_DIM = 64;
unsigned EMBED_DIM = 64;
unsigned SEG_DIM = 32;
unsigned H1DIM = 48;
unsigned H2DIM = 36;
unsigned TAG_DIM = 16;
unsigned TAG_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned DURATION_DIM = 8;

bool eval = false;
cnn::Dict d;
int kNONE;
int kSOW;
int kEOW;

// given the first character of a UTF8 block, find out how wide it is
// see http://en.wikipedia.org/wiki/UTF-8 for more info
inline unsigned int UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else abort();
}

struct PrefixNode {
  PrefixNode() :
      terminal(false),
      bias(nullptr), pred(nullptr), zero_cond(nullptr), zero_child(nullptr),
      one_cond(nullptr), one_child(nullptr) {}

  ~PrefixNode() {
    delete zero_child;
    delete one_child;
  }

  bool terminal;
  Parameters* bias;
  Parameters* pred;
  Parameters* zero_cond;
  PrefixNode* zero_child;
  Parameters* one_cond;
  PrefixNode* one_child;
};

// instructions for use
//   1) add all codes using add("0000") and the like
//   2) then call AllocateParameters(m, dim)
struct PrefixCode {
  PrefixCode() : params_allocated(false) {}

  PrefixNode* add(const string& pfc) {
    assert(!params_allocated);
    PrefixNode* cur = &root;
    for (unsigned i = 0; i < pfc.size(); ++i) {
      if (cur->terminal) {
        cerr << "Prefix property violated at position " << i << " of " << pfc << endl;
        abort();
      }
      assert(pfc[i] == '0' || pfc[i] == '1');
      PrefixNode*& next = pfc[i] == '0' ? cur->zero_child : cur->one_child;
      if (!next) next = new PrefixNode;
      cur = next;
    }
    cur->terminal = true;
    return cur;
  }

  void AllocateParameters_rec(Model& m, unsigned dim, PrefixNode* n) {
    if (!n->terminal) {
      if (!n->zero_child || !n->one_child) {
        cerr << "Non-binary production in prefix code\n";
        abort();
      }
      n->bias = m.add_parameters({1});
      n->pred = m.add_parameters({dim});
      n->zero_cond = m.add_parameters({dim});
      AllocateParameters_rec(m, dim, n->zero_child);
      n->one_cond = m.add_parameters({dim});
      AllocateParameters_rec(m, dim, n->one_child);
    }
  }

  void AllocateParameters(Model& m, unsigned dim) {
    params_allocated = true;
    AllocateParameters_rec(m, dim, &root);
  }

  bool params_allocated;
  PrefixNode root;
};

// returns embeddings of labels
struct SymbolEmbedding {
  SymbolEmbedding(Model& m, unsigned n, unsigned dim) {
    p_labels = m.add_lookup_parameters(n, {dim});
  }
  void new_graph(ComputationGraph& g) { cg = &g; }
  Expression embed(unsigned label_id) {
    return lookup(*cg, p_labels, label_id);
  }
  ComputationGraph* cg;
  LookupParameters* p_labels;
};

struct PrefixCodeDecoder {
  LSTMBuilder decoder;
  PrefixCode* pfc;
  Parameters* p_start;
  explicit PrefixCodeDecoder(Model& model, PrefixCode* pc) :
      decoder(LAYERS, CHAR_DIM, EMBED_DIM, &model), pfc(pc) {
    p_start = model.add_parameters({EMBED_DIM});
  }
  Expression loss(ComputationGraph& cg, const Expression& v, const string& code) {
    decoder.new_graph(cg);
    Expression h = tanh(v);
    vector<Expression> init = {v, h};
    decoder.start_new_sequence(init);
    Expression start = parameter(cg, p_start);
    PrefixNode* cur = &pfc->root;
    decoder.add_input(start);
    size_t i = 0;
    vector<Expression> errs(code.size());
    while(i < code.size()) {
      assert(cur);
      Expression pred = decoder.back();
      Expression rp = parameter(cg, cur->pred);
      Expression bias = parameter(cg, cur->bias);
      Expression p = logistic(dot_product(pred, rp) + bias);
      // maybe squared error instead of xentropy?
      if (code[i] == '0') p = 1.f - p;
      errs[i] = log(p);
      Expression cond = parameter(cg, code[i] == '0' ? cur->zero_cond : cur->one_cond);
      decoder.add_input(cond);
      cur = code[i] == '0' ? cur->zero_child : cur->one_child;
      ++i;
    }
    assert(cur->terminal);
    return -sum(errs);
  }
};

template <class Builder>
struct BiCharLSTM {
  Builder l2rbuilder;
  Builder r2lbuilder;
  Parameters* p_f2c;
  Parameters* p_r2c;
  Parameters* p_cb;
  Parameters* p_c2x;
  Parameters* p_xb;
  SymbolEmbedding sym;

  explicit BiCharLSTM(Model& model) :
      l2rbuilder(LAYERS, CHAR_DIM, EMBED_DIM, &model),
      r2lbuilder(LAYERS, CHAR_DIM, EMBED_DIM, &model),
      sym(model, d.size(), CHAR_DIM) {
    p_f2c = model.add_parameters({EMBED_DIM, CHAR_DIM});
    p_r2c = model.add_parameters({EMBED_DIM, CHAR_DIM});
    p_cb = model.add_parameters({EMBED_DIM});
    p_c2x = model.add_parameters({EMBED_DIM, EMBED_DIM});
    p_xb = model.add_parameters({EMBED_DIM});
  }

  Expression embed(ComputationGraph& cg, const vector<unsigned>& x) {
    l2rbuilder.new_graph(cg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);
    r2lbuilder.start_new_sequence();
    sym.new_graph(cg);
    Expression f2c = parameter(cg, p_f2c);
    Expression r2c = parameter(cg, p_r2c);
    Expression cb = parameter(cg, p_cb);
    Expression c2x = parameter(cg, p_c2x);
    Expression xb = parameter(cg, p_xb);

    int len = x.size();
    vector<Expression> xe(len + 2);
    xe[0] = sym.embed(kSOW);
    for (int i = 0; i < len; ++i)
      xe[i+1] = sym.embed(x[i]);
    xe.back() = sym.embed(kEOW);

    len += 2;
    for (int i = 0; i < len; ++i)
      l2rbuilder.add_input(xe[i]);
    for (int i = len - 1; i >= 0; --i)
      r2lbuilder.add_input(xe[i]);
    Expression c = rectify(affine_transform({cb, f2c, l2rbuilder.back(), r2c, r2lbuilder.back()}));
    return affine_transform({xb, c2x, c});
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  Model model;
  Trainer* sgd = nullptr;
  sgd = new SimpleSGDTrainer(&model);
  vector<pair<string,vector<unsigned>>> training;
  kSOW = d.Convert("<w>");
  kEOW = d.Convert("</w>");
  PrefixCode pc;
  {
    cerr << "Reading training data from " << argv[1] << " ...\n";
    ifstream in(argv[1]);
    string line;
    string code;
    vector<unsigned> chars;
    while(getline(in, line)) {
      size_t s1 = line.find('\t');
      size_t s2 = line.rfind('\t');
      if (s1 == s2 || s2 == (s1+1) || s1 == string::npos) {
        cerr << "malformed input: " << line << endl;
        abort();
      }
      code = line.substr(s1 + 1, s2 - s1 - 1);
      pc.add(code);
      size_t cur = s2 + 1;
      chars.clear();
      while(cur < line.size()) {
        size_t len = UTF8Len(line[cur]);
        chars.push_back(d.Convert(line.substr(cur, len)));
        cur += len;
      }
      training.push_back(make_pair(code, chars));
    }
  }
  cerr << "Character set size = " << d.size() << endl;
  pc.AllocateParameters(model, CODE_DIM);
  BiCharLSTM<LSTMBuilder> bclm(model);
  PrefixCodeDecoder d(model, &pc);
  cerr << "Parameters allocated.\n";
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  unsigned report_every_i = 50;
  unsigned si = training.size();
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
      Expression w = bclm.embed(cg, sent.second);
      d.loss(cg, w, sent.first);
      ttags += 1;
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update(1.0);
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / ttags) << ") ";
  }
  delete sgd;
}

