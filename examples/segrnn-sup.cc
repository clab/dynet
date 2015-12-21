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
#include <set>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 1;
unsigned INPUT_DIM = 64;
unsigned XCRIBE_DIM = 64;
unsigned SEG_DIM = 48;
unsigned H1DIM = 64;
unsigned H2DIM = 48;
unsigned TAG_DIM = 16;
unsigned TAG_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned DURATION_DIM = 4;

bool eval = false;
cnn::Dict d;
cnn::Dict td;
int kNONE;
int kSOS;
int kEOS;

bool use_pretrained_embeding = true;

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

struct DurationEmbedding {
  virtual ~DurationEmbedding();
  virtual void new_graph(ComputationGraph& g) = 0;
  virtual Expression embed(unsigned dur) = 0;
};

DurationEmbedding::~DurationEmbedding() {}

struct MLPDurationEmbedding : public DurationEmbedding {
  MLPDurationEmbedding(Model& m, unsigned hidden, unsigned dim) {
    p_zero = m.add_parameters({dim});
    p_d2h = m.add_parameters({hidden, 2});
    p_hb = m.add_parameters({hidden});
    p_h2o = m.add_parameters({dim, hidden});
    p_ob = m.add_parameters({dim});
    dur_xs.resize(10000, vector<float>(2));
    for (unsigned i = 1; i < dur_xs.size(); ++i) {
      dur_xs[i][0] = i;
      dur_xs[i][1] = float(i) / logf(2);
    }
  }
  void new_graph(ComputationGraph& g) override {
    zero = parameter(g, p_zero);
    d2h = parameter(g, p_d2h);
    hb = parameter(g, p_hb);
    h2o = parameter(g, p_h2o);
    ob = parameter(g, p_ob);
    cg = &g;
  }
  Expression embed(unsigned dur) override {
    if (dur) {
      Expression x = input(*cg, {2}, &dur_xs[dur]);
      Expression h = rectify(affine_transform({hb, d2h, x}));
      return affine_transform({ob, h2o, h});
    } else {
      return zero;
    }
  }
 
  ComputationGraph* cg;
  vector<vector<float>> dur_xs;
  Expression zero, d2h, hb, h2o, ob;
  Parameters* p_zero;
  Parameters* p_d2h;
  Parameters* p_hb;
  Parameters* p_h2o;
  Parameters* p_ob;
};

struct BinnedDurationEmbedding : public DurationEmbedding {
  BinnedDurationEmbedding(Model& m, unsigned dim, unsigned num_bins = 8) : max_bin(num_bins - 1) {
    assert(num_bins > 0);
    p_e = m.add_lookup_parameters(num_bins, {dim});
  }
  void new_graph(ComputationGraph& g) override {
    cg = &g;
  }
  Expression embed(unsigned dur) override {
    if (dur) dur = static_cast<unsigned>(log(dur) / log(1.6)) + 1;
    if (dur > max_bin) { dur = max_bin; }
    return lookup(*cg, p_e, dur);
  }
  ComputationGraph* cg;
  unsigned max_bin;
  LookupParameters* p_e;
};

template <class Builder>
struct BiTrans {
  Builder l2rbuilder;
  Builder r2lbuilder;
  Parameters* p_f2c;
  Parameters* p_r2c;
  Parameters* p_cb;

  explicit BiTrans(Model& model) :
      l2rbuilder(LAYERS, INPUT_DIM, XCRIBE_DIM, &model),
      r2lbuilder(LAYERS, INPUT_DIM, XCRIBE_DIM, &model) {
    p_f2c = model.add_parameters({XCRIBE_DIM, INPUT_DIM});
    p_r2c = model.add_parameters({XCRIBE_DIM, INPUT_DIM});
    p_cb = model.add_parameters({XCRIBE_DIM});
  }

  vector<Expression> transcribe(ComputationGraph& cg, const vector<Expression>& x) {
    l2rbuilder.new_graph(cg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);
    r2lbuilder.start_new_sequence();
    Expression f2c = parameter(cg, p_f2c);
    Expression r2c = parameter(cg, p_r2c);
    Expression cb = parameter(cg, p_cb);

    const int len = x.size();
    vector<Expression> fwd(len), rev(len), res(len);
    for (int i = 0; i < len; ++i)
      fwd[i] = l2rbuilder.add_input(x[i]);
    for (int i = len - 1; i >= 0; --i)
      rev[i] = r2lbuilder.add_input(x[i]);
    for (int i = 0; i < len; ++i)
      res[i] = affine_transform({cb, f2c, fwd[i], r2c, rev[i]});
    return res;
  }
};

// uni-directional segment embeddings
// first call construct_chart(sequence), then access the embeddings of the spans with operator()(i,j)
template <class Builder>
struct SegEmbedUni {
  Parameters* p_h0;
  int len;
  Builder builder;
  vector<vector<Expression>> h;  // h[i][length of segment - 1]
  explicit SegEmbedUni(Model& m) :
      builder(LAYERS, XCRIBE_DIM, SEG_DIM, &m) {
    p_h0 = m.add_parameters({XCRIBE_DIM});
  }
  void construct_chart(ComputationGraph& cg, const vector<Expression>& c, int max_seg_len = 0) {
    len = c.size();
    h.clear();
    h.resize(len);
    Expression h0 = parameter(cg, p_h0);
    builder.new_graph(cg);
    for (int i = 0; i < len; ++i) {
      int max_j = i + len;
      if (max_seg_len) max_j = i + max_seg_len;
      if (max_j > len) max_j = len;
      int seg_len = max_j - i;
      auto& hi = h[i];
      hi.resize(seg_len);

      builder.start_new_sequence();
      builder.add_input(h0);
      for (int k = 0; k < seg_len; ++k)
        hi[k] = builder.add_input(c[i+k]);
    }
  }
  // returns the embedding of segment [i,j]
  const Expression& operator()(int i, int j) const {
    assert(j < len);
    assert(j >= i);
    return h[i][j - i];
  }
};

// uni-directional segment embeddings
// first call construct_chart(sequence), then access the embeddings of the spans with operator()(i,j)
template <class Builder>
struct SegEmbedBi {
  int len;
  vector<vector<pair<Expression, Expression>>> h;
  SegEmbedUni<Builder> fwd, rev;
  explicit SegEmbedBi(Model& m) : fwd(m), rev(m) {}
  void construct_chart(ComputationGraph& cg, const vector<Expression>& c, int max_seg_len = 0) {
    len = c.size();
    fwd.construct_chart(cg, c, max_seg_len);
    vector<Expression> rc(len);
    for (int i = 0; i < len; ++i) rc[i] = c[len - 1 - i];
    rev.construct_chart(cg, rc, max_seg_len);
    h.clear();
    h.resize(len);
    for (int i = 0; i < len; ++i) {
      int max_j = i + len;
      if (max_seg_len) max_j = i + max_seg_len;
      if (max_j > len) max_j = len;
      int seg_len = max_j - i;
      auto& hi = h[i];
      hi.resize(seg_len);
      for (int k = 0; k < seg_len; ++k) {
        int j = i + k;
        const Expression& fe = fwd(i, j);
        const Expression& re = rev(len-1-j, len-1-i);
        hi[k] = make_pair(fe,re);
      }
    }
  }
  const pair<Expression, Expression>& operator()(int i, int j) const {
    assert(j >= i);
    assert(j < len);
    return h[i][j-i];
  }
};

template <class Builder>
struct ZerothOrderSemiCRF {
  SymbolEmbedding* xe;
  SymbolEmbedding* ye;
  DurationEmbedding* de;
  BiTrans<Builder> bt;
  SegEmbedBi<Builder> seb;
  explicit ZerothOrderSemiCRF(Model& model) :
      bt(model), seb(model) {
    xe = new SymbolEmbedding(model, VOCAB_SIZE, INPUT_DIM);

    if (use_pretrained_embeding) {
       ifstream fin("/usr1/corpora/LDC2011T13/vec-cwin.txt");  
       string s;
       while( getline(fin,s) )
       {   
        vector <string> fields;
        boost::algorithm::trim(s);
        boost::algorithm::split( fields, s, boost::algorithm::is_any_of( " " ) );
        string word = fields[0];
        vector<float> p_embeding;
        for (int ind = 1; ind < fields.size(); ++ind){
          p_embeding.push_back(std::stod(fields[ind]));
        }
        if (d.Contains(word)){
          // cout << "init" << endl;
          xe->p_labels->Initialize(d.Convert(word), p_embeding);
        }

      }
    }

    ye = new SymbolEmbedding(model, TAG_SIZE, TAG_DIM);
    de = new BinnedDurationEmbedding(model, DURATION_DIM);

    // potential lower layer
    p_d2h1 = model.add_parameters({H1DIM, DURATION_DIM});
    p_y2h1 = model.add_parameters({H1DIM, TAG_DIM});
    p_fwd2h1 = model.add_parameters({H1DIM, SEG_DIM});
    p_rev2h1 = model.add_parameters({H1DIM, SEG_DIM});
    p_h1b = model.add_parameters({H1DIM});

    // potential upper layer
    p_h12h2 = model.add_parameters({H2DIM, H1DIM});
    p_h2b = model.add_parameters({H2DIM});

    // potential output layer
    p_h22o = model.add_parameters({1, H2DIM});
    p_ob = model.add_parameters({1});
  }

  // return Expression of total loss
  Expression SupervisedLoss(const vector<int>& x,
                            const vector<pair<int,int>>& yz,  // .first = y, .second = duration (z)
                            ComputationGraph& cg,
                            int max_seg_len = 0) {
    int len = x.size();

    // is_ref[i][j][y] returns true iff, in the reference, span(i,j) is labeled y
    vector<vector<vector<bool>>> is_ref(len, vector<vector<bool>>(len+1, vector<bool>(TAG_SIZE, false)));
    unsigned cur = 0;
    for (unsigned ri = 0; ri < yz.size(); ++ri) {
      assert(cur < len);
      int y = yz[ri].first;
      int dur = yz[ri].second;
      if (max_seg_len && dur > max_seg_len) {
        cerr << "max_seg_len=" << max_seg_len << " but reference duration is " << dur << endl;
        abort();
      }
      unsigned j = cur + dur;
      assert(j <= len);
      is_ref[cur][j][y] = true;
      // cerr << "Span[" << cur << "," << j << ")=" << td.Convert(y) << endl;
      cur = j;
    }
    assert(cur == len);

    xe->new_graph(cg);
    ye->new_graph(cg);
    de->new_graph(cg);
    Expression d2h1 = parameter(cg, p_d2h1);
    Expression y2h1 = parameter(cg, p_y2h1);
    Expression fwd2h1 = parameter(cg, p_fwd2h1);
    Expression rev2h1 = parameter(cg, p_rev2h1);
    Expression h1b = parameter(cg, p_h1b);
    Expression h12h2 = parameter(cg, p_h12h2);
    Expression h2b = parameter(cg, p_h2b);
    Expression h22o = parameter(cg, p_h22o);
    Expression ob = parameter(cg, p_ob);
    vector<Expression> xins(x.size());
    for (int i = 0; i < len; ++i)
      xins[i] = xe->embed(x[i]);
    vector<Expression> c = bt.transcribe(cg, xins);
    seb.construct_chart(cg, c, max_seg_len);
    vector<Expression> fwd(len+1);  // fwd trellis for model
    vector<Expression> ref_fwd(len+1); // fwd trellis for reference
    vector<Expression> f, fr;

    // careful: in the other algorithms spans are [i,j], here they are [i,j)
    for (int j = 1; j <= len; ++j) {
      // fwd[j] is the total unnoramlized probability for all segmentations / labels
      // ending (and including) the symbol at position j-1
      f.clear(); // f stores all additive contributions to item [j]
      fr.clear();
      for (unsigned tag = 0; tag < TAG_SIZE; ++tag) {
        Expression y = ye->embed(tag);
        const int i_start = max_seg_len ? max(0, j - max_seg_len) : 0;
        for (int i = i_start; i < j; ++i) {  // i is the starting position
          bool matches_ref = is_ref[i][j][tag];
          auto seg_embedding_ij = seb(i, j-1); // pair<expr, expr>
          Expression d = de->embed(j - i);
          // factor includes: fwd embedding, rev embedding, duration embedding, label embedding
          Expression h1 = rectify(affine_transform({h1b, d2h1, d, y2h1, y, fwd2h1, seg_embedding_ij.first, rev2h1, seg_embedding_ij.second}));
          Expression h2 = tanh(affine_transform({h2b, h12h2, h1}));
          //Expression p = exp(affine_transform({ob, h22o, h2}));
          Expression p = affine_transform({ob, h22o, h2});
          if (i == 0) { // fwd[0] is the path up and including -1, so it's the empty set, i.e., its probability is 1
            f.push_back(p);
            if (matches_ref) fr.push_back(p);
          } else {
            // f.push_back(p * fwd[i]);
            // if (matches_ref) fr.push_back(p * ref_fwd[i]);
            f.push_back(p + fwd[i]);
            if (matches_ref) fr.push_back(p + ref_fwd[i]);
          }
        }
      }
      // cerr << "info\t" << j << "\tsize: " << f.size() << endl;
      // for(auto& e : f){
      //   cerr << "node\t" << as_scalar(cg.get_value(e.i)) << endl;
      // }
      // cerr << "size of vector f at j = " << j << " equals " << f.size() << endl;
      // cerr << "info end" << endl;
      fwd[j] = logsumexp(f);

      // cerr << "shock me\t" << as_scalar(cg.get_value((fwd[j]))) << endl;

      if (fr.size()) ref_fwd[j] = logsumexp(fr);
    }
    //return log(fwd.back()) - log(ref_fwd.back());
    return fwd.back() - ref_fwd.back();
  }

  Expression ViterbiDecode(const vector<int>& x,
                            const vector<pair<int,int>>& yz_gold,  // .first = y, .second = duration (z)
                            ComputationGraph& cg,
                            vector<pair<int, int>> &yz_pred,
                            int max_seg_len = 0) {
    yz_pred.clear();
    int len = x.size();

    xe->new_graph(cg);
    ye->new_graph(cg);
    de->new_graph(cg);
    Expression d2h1 = parameter(cg, p_d2h1);
    Expression y2h1 = parameter(cg, p_y2h1);
    Expression fwd2h1 = parameter(cg, p_fwd2h1);
    Expression rev2h1 = parameter(cg, p_rev2h1);
    Expression h1b = parameter(cg, p_h1b);
    Expression h12h2 = parameter(cg, p_h12h2);
    Expression h2b = parameter(cg, p_h2b);
    Expression h22o = parameter(cg, p_h22o);
    Expression ob = parameter(cg, p_ob);
    vector<Expression> xins(x.size());
    for (int i = 0; i < len; ++i)
      xins[i] = xe->embed(x[i]);
    vector<Expression> c = bt.transcribe(cg, xins);
    seb.construct_chart(cg, c, max_seg_len);
    vector<Expression> fwd(len+1);  // fwd trellis for model
    vector<Expression> f, fr;
    vector<std::tuple<int, int, unsigned>> ijt; // ijt stores positions where we get the max (i.e. i, j and tag)
    vector<std::tuple<int, int, unsigned>> it; // it stores positions where we get the max ending at j
    it.push_back(make_tuple(0,0,0)); // push one to make the index consistent, now it[j] means j rather than j+1
    // careful: in the other algorithms spans are [i,j], here they are [i,j)
    for (int j = 1; j <= len; ++j) {
      // fwd[j] is the total unnoramlized probability for all segmentations / labels
      // ending (and including) the symbol at position j-1
      f.clear(); // f stores all additive contributions to item [j]
      fr.clear();
      ijt.clear();
      for (unsigned tag = 0; tag < TAG_SIZE; ++tag) {
        Expression y = ye->embed(tag);
        const int i_start = max_seg_len ? max(0, j - max_seg_len) : 0;
        for (int i = i_start; i < j; ++i) {  // i is the starting position
          auto seg_embedding_ij = seb(i, j-1); // pair<expr, expr>
          Expression d = de->embed(j - i);
          // factor includes: fwd embedding, rev embedding, duration embedding, label embedding
          Expression h1 = rectify(affine_transform({h1b, d2h1, d, y2h1, y, fwd2h1, seg_embedding_ij.first, rev2h1, seg_embedding_ij.second}));
          Expression h2 = tanh(affine_transform({h2b, h12h2, h1}));
          //Expression p = exp(affine_transform({ob, h22o, h2}));
          Expression p = affine_transform({ob, h22o, h2});

          if (i == 0) { // fwd[0] is the path up and including -1, so it's the empty set, i.e., its probability is 1
            f.push_back(p);
          } else {
            f.push_back(p + fwd[i]);
          }
          ijt.push_back(std::make_tuple(i, j, tag));
        }
      }
      // cerr << "size of vector f at j = " << j << " equals " << f.size() << endl;
      unsigned max_ind = 0;
      auto max_val = as_scalar(cg.get_value(f[0].i));
      for(auto ind = 1; ind < f.size(); ++ind){
        auto val = as_scalar(cg.get_value(f[ind].i));
        // cerr << val << endl;
        if (max_val < val){
          max_val = val;
          max_ind = ind;
        }
      }
      fwd[j] = f[max_ind];
      it.push_back(ijt[max_ind]);
      // cerr << "max value = \t" << as_scalar(cg.get_value(fwd[j])) << endl;
    }
    // for(auto j = 1; j <= len; ++j) {
    //   cerr << j << "\t" << std::get<0>(it[j]) << "\t" << std::get<1>(it[j]) << "\t" << std::get<2>(it[j]) << endl;
    // }
    auto cur_j = len;
    vector<std::tuple<int, int, unsigned>> pred;
    while(cur_j > 0){
      auto cur_i = std::get<0>(it[cur_j]);
      pred.push_back(make_tuple(cur_i, cur_j, std::get<2>(it[cur_j])));
      yz_pred.push_back(make_pair(std::get<2>(it[cur_j]), cur_j - cur_i));
      cur_j = cur_i;
    }
    std::reverse(pred.begin(),pred.end());
    std::reverse(yz_pred.begin(),yz_pred.end());
    cout << "==pred==" << endl;
    // for(auto e : pred){
    //   cout << std::get<0>(e) << "\t" << std::get<1>(e) << "\t" << std::get<2>(e) << endl;
    // }

    for(auto e: yz_pred){
      cout << e.first << "," << e.second << "\t";
    }
    cout << endl;

    cout << "==gold==" << endl;
    for(auto e: yz_gold){
      cout << e.first << "," << e.second << "\t";
    }
    cout << endl;

    // abort();
    return fwd.back();
  }
  

  


  Parameters* p_d2h1, *p_y2h1, *p_fwd2h1, *p_rev2h1, *p_h1b;
  Parameters* p_h12h2, *p_h2b;
  Parameters* p_h22o, *p_ob;
};

// a a 0 1 a ||| O:1 O:1 N:2 O:1
pair<vector<int>,vector<pair<int,int>>> ParseTrainingInstance(const std::string& line) {
  std::istringstream in(line);
  std::string word;
  std::string sep = "|||";
  vector<int> x;
  vector<pair<int,int>> yz;
  while(1) {
    in >> word;
    if (word == sep) break;
    x.push_back(d.Convert(word));
  }
  while(1) {
    in >> word;
    if (!in) break;
    size_t p = word.rfind(':');
    if (p == string::npos || p == 0 || p == (word.size() - 1)) {
      cerr << "mal-formed label: " << word << endl;
      abort();
    }
    int y = td.Convert(word.substr(0, p));
    int z = atoi(word.substr(p+1).c_str());
    yz.push_back(make_pair(y,z));
  }
  return make_pair(x, yz);
}

bool inline check_max_seg(const vector<pair<int,int>>& yz, int max_seg_len = 0){
  if (max_seg_len == 0) return true;
  for (unsigned ri = 0; ri < yz.size(); ++ri) {
    int dur = yz[ri].second;
    if (max_seg_len && dur > max_seg_len) {
      cerr << "SKIP: max_seg_len=" << max_seg_len << " but reference duration is " << dur << endl;
      return false;
    }
  }
  return true;
}

double compute_f_score(vector<vector<pair<int,int>>>& yz_preds, vector<vector<pair<int,int>>>& yz_golds){
  assert(yz_preds.size() == yz_golds.size());
  int p_correct = 0;
  int r_correct = 0;
  int p_total = 0; 
  int r_total = 0;
  for (unsigned i = 0; i < yz_preds.size(); i++){
    // for sentence i
    std::set<pair<int,int>> gold;
    std::set<pair<int,int>> pred;
    vector<pair<int,int>>& yz_pred = yz_preds[i];
    vector<pair<int,int>>& yz_gold = yz_golds[i];
    int cur = 0;
    for(auto e : yz_pred){
      pred.insert(make_pair(cur, cur + e.second));
      cur += e.second;
    }
    cur = 0;
    for(auto e : yz_gold){
      gold.insert(make_pair(cur, cur + e.second));
      cur += e.second;
    }

    for (auto e : pred){
      if(gold.find(e) != gold.end()){
        p_correct++;
      }
      p_total++;
    }
    for (auto e : gold){
      if(pred.find(e) != pred.end()){
        r_correct++;
      }
      r_total++;
    }
  }
  double p = (double)(p_correct) / (double)(p_total);
  double r = (double)(r_correct) / (double)(r_total);
  double f = 2.0 * ((p * r) / (p + r));
  cerr << "p: " << p << "\tr: " << r << "\tf: " << f << endl;
  return f; 

}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  vector<pair<vector<int>,vector<pair<int,int>>>> training, dev;
  string line;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      training.push_back(ParseTrainingInstance(line));
    }
  }
  d.Freeze();  // no new word types allowed
  td.Freeze(); // no new tag types allowed

  d.SetUnk("<UNK>"); // set UNK to allow the unseen character in the dev and test set

  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      dev.push_back(ParseTrainingInstance(line));
    }
  }
  VOCAB_SIZE = d.size();
  TAG_SIZE = td.size();

  cerr << "TAG_SIZE = " << td.size() << endl;
  ostringstream os;
  os << "0scrf"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << XCRIBE_DIM
     << "-pid" << getpid();
  const string fname = os.str() + ".params";
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  const string f_d_name = os.str() + ".dict";
  ofstream out_d(f_d_name);
  boost::archive::text_oarchive oa_d(out_d);
  oa_d << d;

  const string f_td_name = os.str() + ".tdict";
  ofstream out_td(f_td_name);
  boost::archive::text_oarchive oa_td(out_td);
  oa_td << td;

  Model model;
  auto sgd = new SimpleSGDTrainer(&model);
  int max_seg_len = 10;

  ZerothOrderSemiCRF<LSTMBuilder> crf(model);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  unsigned report_every_i = 10;
  unsigned dev_every_i_reports = 100;
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
        // shuffle(order.begin(), order.end(), *rndeng);
      }

      // build graph for this instance
      ComputationGraph cg;
      auto& sent = training[order[si]];
      ++si;
      if(check_max_seg(sent.second, max_seg_len)){
        crf.SupervisedLoss(sent.first, sent.second, cg, max_seg_len);
        ttags += sent.second.size();
        loss += as_scalar(cg.forward());
        cg.backward();
        sgd->update(1.0);
      }
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / ttags) << ") ";
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dtags = 0;
      double f_best = 0;
      vector<vector<pair<int,int>>> yz_preds;
      vector<vector<pair<int,int>>> yz_golds;
      for (auto& sent : dev) {
        ComputationGraph cg;
        if(check_max_seg(sent.second, max_seg_len)){
          crf.SupervisedLoss(sent.first, sent.second, cg, max_seg_len);
          dtags += sent.second.size();
          dloss += as_scalar(cg.forward());
        }
        vector<pair<int, int>> yz_pred;
        crf.ViterbiDecode(sent.first, sent.second, cg, yz_pred, max_seg_len);
        yz_golds.push_back(sent.second);
        yz_preds.push_back(yz_pred);
      }
      cout << "iter_end" << endl;

      double f = compute_f_score(yz_preds, yz_golds);

      if (f > f_best) {
        f_best = f;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << ' ';
    }
  }
  delete sgd;
}

