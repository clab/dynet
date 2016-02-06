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
#include <boost/program_options.hpp>


using namespace std;
using namespace cnn;

namespace po = boost::program_options;

struct PKey {
  PKey(int x1, int x2, unsigned x3)
    : t(x1,x2,x3) {}
  tuple<int, int, unsigned> t;
};

bool operator==(const PKey &a, const PKey &b)
{
  return std::get<0>(a.t) == std::get<0>(b.t) &&
         std::get<1>(a.t) == std::get<1>(b.t) &&
         std::get<2>(a.t) == std::get<2>(b.t);
}

namespace std {
  template <>
  struct hash<PKey>
  {
    std::size_t operator()(const PKey& e) const
    {
      return (std::get<0>(e.t)) | (std::get<1>(e.t) << 16) | (std::get<2>(e.t) << 24);
    }
  };

}

unsigned LAYERS = 1;
unsigned INPUT_DIM = 64;
unsigned XCRIBE_DIM = 32;
unsigned SEG_DIM = 16;
unsigned H1DIM = 32;
unsigned H2DIM = 32;
unsigned TAG_DIM = 32;
unsigned DURATION_DIM = 4;

unsigned int DATA_MAX_SEG_LEN = 0;

bool use_pretrained_embeding = false;
bool use_dropout = false;
float dropout_rate; 
bool ner_tagging = false;
string pretrained_embeding = "";

// returns embeddings of labels
struct SymbolEmbedding {
  SymbolEmbedding(Model& m, unsigned n, unsigned dim) {
    p_labels = m.add_lookup_parameters(n, {dim});
  }
  void load_embedding(cnn::Dict& d, string pretrain_path){
    ifstream fin(pretrain_path);  
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
          p_labels->Initialize(d.Convert(word), p_embeding);
        }
      }
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
  int max_bin;
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
    p_f2c = model.add_parameters({XCRIBE_DIM, XCRIBE_DIM});
    p_r2c = model.add_parameters({XCRIBE_DIM, XCRIBE_DIM});
    p_cb = model.add_parameters({XCRIBE_DIM});
  }

  vector<Expression> transcribe(ComputationGraph& cg, const vector<Expression>& x) {
    l2rbuilder.new_graph(cg);
    if(use_dropout){
      l2rbuilder.set_dropout(dropout_rate);
    }else{
      l2rbuilder.disable_dropout();
    }
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);
    if(use_dropout){
      r2lbuilder.set_dropout(dropout_rate);
    }else{
      r2lbuilder.disable_dropout();
    }
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
    if(use_dropout){
      builder.set_dropout(dropout_rate);
    }else{
      builder.disable_dropout();
    }
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
struct SegmentalRNN {
  SymbolEmbedding* xe;
  SymbolEmbedding* ye;
  DurationEmbedding* de;
  BiTrans<Builder> bt;
  SegEmbedBi<Builder> seb;
  cnn::Dict d;
  cnn::Dict td;
  explicit SegmentalRNN(Model& model, cnn::Dict& d_, cnn::Dict& td_) :
      bt(model), seb(model) {
    d = d_;
    td = td_;
    xe = new SymbolEmbedding(model, d.size(), INPUT_DIM);
    if (use_pretrained_embeding) {
       xe->load_embedding(d, pretrained_embeding);
    }
    ye = new SymbolEmbedding(model, td.size(), TAG_DIM);
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

    // context aware
    p_c_start = model.add_parameters({XCRIBE_DIM});
    p_c_end = model.add_parameters({XCRIBE_DIM});
    p_cf2h1 = model.add_parameters({H1DIM, XCRIBE_DIM});
    p_ce2h1 = model.add_parameters({H1DIM, XCRIBE_DIM});
  }

  // Adapted the ConstructInput to apply the SegmentRNNs to different inputs
  vector<Expression> ConstructInput(const vector<int>& x,
                                  ComputationGraph& cg){
  xe->new_graph(cg);
  vector<Expression> xins(x.size());
  for (int i = 0; i < x.size(); ++i)
    xins[i] = xe->embed(x[i]);
    return xins;
  }

  unordered_map<PKey,Expression> ConstructSegmentMap(vector<Expression>& xins,
                                                     ComputationGraph& cg,
                                                     int max_seg_len = 0)
  {
    unordered_map<PKey, Expression> p_map;
    int len = xins.size();

    // context aware
    Expression c_start = parameter(cg, p_c_start);
    Expression c_end = parameter(cg, p_c_end);
    Expression cf2h1 = parameter(cg, p_cf2h1);
    Expression ce2h1 = parameter(cg, p_ce2h1);

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

    vector<Expression> c = bt.transcribe(cg, xins);
    seb.construct_chart(cg, c, max_seg_len);

    // careful: in the other algorithms spans are [i,j], here they are [i,j)
    for (int j = 1; j <= len; ++j) {
      for (unsigned tag = 0; tag < td.size(); ++tag) {
        Expression y = ye->embed(tag);
        const int i_start = max_seg_len ? max(0, j - max_seg_len) : 0;
        for (int i = i_start; i < j; ++i) {  // i is the starting position
          PKey key_tuple(i,j,tag);
          auto seg_embedding_ij = seb(i, j-1); // pair<expr, expr>
          Expression d = de->embed(j - i);
          // To be context aware, add c_{i-1} and c_{j}
          Expression cf_embeding = (i == 0) ? c_start : c[i-1];
          Expression ce_embeding = (j == len) ? c_end : c[j];

          Expression h1 = rectify(affine_transform({h1b, d2h1, d, y2h1, y, fwd2h1, seg_embedding_ij.first, rev2h1, seg_embedding_ij.second, cf2h1, cf_embeding, ce2h1, ce_embeding}));
          Expression h2;
          if(use_dropout){
            h2 = dropout(tanh(affine_transform({h2b, h12h2, h1})), dropout_rate);
          }else{
            h2 = tanh(affine_transform({h2b, h12h2, h1}));
          }
          Expression p = affine_transform({ob, h22o, h2});
          p_map[key_tuple] = p;
        }
      }
    }
    return p_map;

  }

  // return Expression of total loss
  Expression SupervisedCRFLoss(vector<Expression>& xins,
                            const vector<pair<int,int>>& yz,  // .first = y, .second = duration (z)
                            ComputationGraph& cg,
                            int max_seg_len = 0) {
    int len = xins.size();
    unordered_map<PKey, Expression> p_map = ConstructSegmentMap(xins, cg, max_seg_len);
    // is_ref[i][j][y] returns true iff, in the reference, span(i,j) is labeled y
    vector<vector<vector<bool>>> is_ref(len, vector<vector<bool>>(len+1, vector<bool>(td.size(), false)));
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

    vector<Expression> fwd(len+1);  // fwd trellis for model
    vector<Expression> ref_fwd(len+1); // fwd trellis for reference
    vector<Expression> f, fr;

    // careful: in the other algorithms spans are [i,j], here they are [i,j)
    for (int j = 1; j <= len; ++j) {
      // fwd[j] is the total unnoramlized probability for all segmentations / labels
      // ending (and including) the symbol at position j-1
      f.clear(); // f stores all additive contributions to item [j]
      fr.clear();
      for (unsigned tag = 0; tag < td.size(); ++tag) {
        const int i_start = max_seg_len ? max(0, j - max_seg_len) : 0;
        for (int i = i_start; i < j; ++i) {  // i is the starting position
          bool matches_ref = is_ref[i][j][tag];
          PKey key_tuple(i,j,tag);
          assert(p_map.find (key_tuple) != p_map.end());
          Expression p = p_map.at(key_tuple);
          if (i == 0) { // fwd[0] is the path up and including -1, so it's the empty set, i.e., its probability is 1
            f.push_back(p);
            if (matches_ref) fr.push_back(p);
          } else {
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

  Expression PartiallySupervisedCRFLoss(vector<Expression>& xins,
                            const vector<pair<int,int>>& yz,  // .first = y, .second = duration (z)
                            ComputationGraph& cg,
                            int max_seg_len = 0) {
    
    // I keep the z here although I am not going to use it
    int len = xins.size();
    unordered_map<PKey, Expression> p_map = ConstructSegmentMap(xins, cg, max_seg_len);

    vector<Expression> fwd(len+1);  // fwd trellis for model
    vector<Expression> f;

    vector<vector<Expression>> fwd_iq(len+1);
    for (int ind = 0; ind < fwd_iq.size(); ++ind){
      // if yz.size() == 2, then we have end state q = 2 means already consume 2 labels
      // therefore here, should resize to yz.size() + 1 so that fwd_iq[len][2] is the end state
      fwd_iq[ind].resize(yz.size()+1);
    }

    vector<vector<Expression>> fq;
    // cg.forward();
    // cerr << "here" << endl;
    // cerr << len << endl;

    // careful: in the other algorithms spans are [i,j], here they are [i,j)
    for (int j = 1; j <= len; ++j) {
      // cg.forward();
      // cerr << "j: " << j << endl;
      // fwd[j] is the total unnoramlized probability for all segmentations / labels
      // ending (and including) the symbol at position j-1
      f.clear(); // f stores all additive contributions to item [j]
      fq.clear();
      fq.resize(yz.size() + 1);
      for(auto e : fq){
        e.clear();
      }

      for (unsigned tag = 0; tag < td.size(); ++tag) {
        const int i_start = max_seg_len ? max(0, j - max_seg_len) : 0;
        for (int i = i_start; i < j; ++i) {  // i is the starting position
          PKey key_tuple(i,j,tag);
          assert(p_map.find (key_tuple) != p_map.end());
          Expression p = p_map.at(key_tuple);

          if (i == 0) { // fwd[0] is the path up and including -1, so it's the empty set, i.e., its probability is 1
            f.push_back(p);
          } else {
            f.push_back(p + fwd[i]);
          }
          
          for (int q = 0; (q < yz.size()+1 && q <= j) ; ++q){
            if (i == 0){
              if(tag == yz[q-1].first && q == 1){
                // q can be 0 now and must be 0 now
                // cerr << "j: " << j << " i: " << i << " q: " << q << endl;
                fq[q].push_back(p);
                // cg.forward();
              }
            } else {
              if(tag == yz[q-1].first && q > 1 && q-1 <= i ) {
                // if q == 0, it's not possible because i > 0, i already consume at least one lable
                // how to default value 0
                // note fwd_iq[max_seg_len+1] can't only consume 1 tag
                // basically, (i-1)/max_seg_len < q-1
                if ((i-1)/max_seg_len < q-1){
                  fq[q].push_back(p + fwd_iq[i][q-1]);
                }
                // else this item got cut
              }
            } 
          }
        }
      }
      fwd[j] = logsumexp(f);
      for (int q = 0; q < yz.size()+1 && q <= j; ++q){
        if(fq[q].size() > 0){
          fwd_iq[j][q] = logsumexp(fq[q]);
        }
      }
    }

    return fwd.back() - fwd_iq.back().back();
  }


  Expression SumParts(int len,
                      const vector<pair<int,int>>& yz,
                      unordered_map<PKey, Expression>& p_map,
                      ComputationGraph& cg,
                      int max_seg_len = 0
                      ) {
    vector<Expression> useful_parts;
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

      PKey key_tuple(cur,j,y);
      assert(p_map.find (key_tuple) != p_map.end());
      Expression p = p_map.at(key_tuple);
      useful_parts.push_back(p);
      cur = j;
    }
    assert(cur == len);
    assert(useful_parts.size() > 0);
    Expression s = sum(useful_parts);
    return s;
  }

    // return Expression of total loss
  Expression SupervisedHingeLoss(vector<Expression>& xins,
                            const vector<pair<int,int>>& yz,  // .first = y, .second = duration (z)
                            ComputationGraph& cg,
                            int max_seg_len = 0) {
    int len = xins.size();
    unordered_map<PKey, Expression> p_map = ConstructSegmentMap(xins, cg, max_seg_len);
    cg.forward();
    vector<Expression> tempvec;
    Expression gold_score = SumParts(len, yz, p_map, cg, max_seg_len);
    tempvec.push_back(gold_score);
    vector<pair<int,int>> yz_pred; 
    PureDecode(len, p_map, cg, yz_pred, max_seg_len);
    Expression predict_score = SumParts(len, yz_pred, p_map, cg, max_seg_len);
    tempvec.push_back(predict_score);
    Expression loss = hinge(concatenate(tempvec), (unsigned)0, (float)10);
    return loss;
  }

  void PureDecode(int len,
                  unordered_map<PKey,Expression>& p_map,
                  ComputationGraph& cg,
                  vector<pair<int, int>> &yz_pred,
                  int max_seg_len = 0){
    yz_pred.clear();
    // vector<Expression> fwd(len+1);  // fwd trellis for model
    // vector<Expression> f;
    vector<float> v_fwd(len+1);
    vector<float> v_f;

    vector<tuple<int, int, unsigned>> ijt; // ijt stores positions where we get the max (i.e. i, j and tag)
    vector<tuple<int, int, unsigned>> it; // it stores positions where we get the max ending at j
    it.push_back(make_tuple(0,0,0)); // push one to make the index consistent, now it[j] means j rather than j+1
    // careful: in the other algorithms spans are [i,j], here they are [i,j)
    for (int j = 1; j <= len; ++j) {
      // fwd[j] is the total unnoramlized probability for all segmentations / labels
      // ending (and including) the symbol at position j-1
      // f.clear(); // f stores all additive contributions to item [j]
      v_f.clear();
      ijt.clear();
      for (unsigned tag = 0; tag < td.size(); ++tag) {
        const int i_start = max_seg_len ? max(0, j - max_seg_len) : 0;
        for (int i = i_start; i < j; ++i) {  // i is the starting position
          PKey key_tuple(i,j,tag);
          Expression p = p_map.at(key_tuple);
          float v_p = as_scalar(cg.get_value(p.i));
          if (i == 0) { // fwd[0] is the path up and including -1, so it's the empty set, i.e., its probability is 1
            // f.push_back(p);
            v_f.push_back(v_p);
          } else {
            // f.push_back(p + fwd[i]);
            v_f.push_back(v_p + v_fwd[i]);
          }
          ijt.push_back(make_tuple(i, j, tag));
        }
      }
      // cerr << "size of vector f at j = " << j << " equals " << f.size() << endl;
      unsigned max_ind = 0;
      // auto max_val = as_scalar(cg.get_value(f[0].i));
      auto max_val = v_f[0];
      // for(auto ind = 1; ind < f.size(); ++ind){
      for(auto ind = 1; ind < v_f.size(); ++ind){
        // auto val = as_scalar(cg.get_value(f[ind].i));
        auto val = v_f[ind];
        // cerr << val << endl;
        if (max_val < val){
          max_val = val;
          max_ind = ind;
        }
      }
      // fwd[j] = f[max_ind];
      v_fwd[j] = v_f[max_ind];
      it.push_back(ijt[max_ind]);
      // cerr << "max value = \t" << as_scalar(cg.get_value(fwd[j])) << endl;
    }
    // for(auto j = 1; j <= len; ++j) {
    //   cerr << j << "\t" << std::get<0>(it[j]) << "\t" << std::get<1>(it[j]) << "\t" << std::get<2>(it[j]) << endl;
    // }
    auto cur_j = len;
    vector<tuple<int, int, unsigned>> pred;
    while(cur_j > 0){
      auto cur_i = std::get<0>(it[cur_j]);
      pred.push_back(make_tuple(cur_i, cur_j, std::get<2>(it[cur_j])));
      yz_pred.push_back(make_pair(std::get<2>(it[cur_j]), cur_j - cur_i));
      cur_j = cur_i;
    }
    std::reverse(pred.begin(),pred.end());
    std::reverse(yz_pred.begin(),yz_pred.end());
  }

  void ViterbiDecode(vector<Expression>& xins,
                           const vector<pair<int,int>>& yz_gold,  // .first = y, .second = duration (z)
                           ComputationGraph& cg,
                           vector<pair<int, int>> &yz_pred,
                           int max_seg_len = 0) {
    
    int len = xins.size();
    unordered_map<PKey, Expression> p_map = ConstructSegmentMap(xins, cg, max_seg_len);

    // Compute everything at this step and use them later
    cg.forward();

    PureDecode(len, p_map, cg, yz_pred, max_seg_len);
    return;
  }

  Parameters* p_d2h1, *p_y2h1, *p_fwd2h1, *p_rev2h1, *p_h1b;
  Parameters* p_h12h2, *p_h2b;
  Parameters* p_h22o, *p_ob;
  // used in the context awareness 
  Parameters* p_c_start, *p_c_end;
  Parameters* p_cf2h1, *p_ce2h1;
};

// a a 0 1 a ||| O:1 O:1 N:2 O:1
pair<vector<int>,vector<pair<int,int>>> ParseTrainingInstance(const std::string& line, cnn::Dict& d, cnn::Dict& td, bool test_only = false) {
  std::istringstream in(line);
  std::string word;
  std::string sep = "|||";
  vector<int> x;
  vector<pair<int,int>> yz;
  while(1) {
    in >> word;
    if (!test_only){
      if (word == sep) break;
    }else{
      if (word == sep) break;
      if (!in) break;
    }
    x.push_back(d.Convert(word));
  }
  if(!test_only){
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
      if (z > DATA_MAX_SEG_LEN){
        DATA_MAX_SEG_LEN = z;
      }
      yz.push_back(make_pair(y,z));
    }
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

double evaluate(vector<vector<pair<int,int>>>& yz_preds,
                vector<vector<pair<int,int>>>& yz_golds,
                cnn::Dict& d,
                cnn::Dict& td){
  assert(yz_preds.size() == yz_golds.size());
  int p_correct = 0;
  int r_correct = 0;
  int p_w_t_correct = 0;
  int r_w_t_correct = 0;
  int p_total = 0; 
  int r_total = 0;
  int p_w_t_total = 0;
  int r_w_t_total = 0;
  int tag_o = ner_tagging ? td.Convert("O") : -1;
  for (unsigned int i = 0; i < yz_preds.size(); i++){
    // for sentence i
    std::set<pair<int,int>> gold;
    std::set<pair<int,int>> pred;
    std::set<tuple<int,int,int>> gold_w_tag;
    std::set<tuple<int,int,int>> pred_w_tag;
    vector<pair<int,int>>& yz_pred = yz_preds[i];
    vector<pair<int,int>>& yz_gold = yz_golds[i];
    int cur = 0;
    for(auto e : yz_pred){

      if (ner_tagging && (e.first == tag_o) ){

      }else{
        pred.insert(make_pair(cur, cur + e.second));
        pred_w_tag.insert(make_tuple(e.first, cur, cur + e.second));
      }
      cur += e.second;
    }
    cur = 0;
    for(auto e : yz_gold){
      if (ner_tagging && (e.first == tag_o) ){

      }else{
        gold.insert(make_pair(cur, cur + e.second));
        gold_w_tag.insert(make_tuple(e.first, cur, cur + e.second));
      }
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

    for (auto e : pred_w_tag){
      if(gold_w_tag.find(e) != gold_w_tag.end()){
        p_w_t_correct++;
      }
      p_w_t_total++;
    }
    for (auto e : gold_w_tag){
      if(pred_w_tag.find(e) != pred_w_tag.end()){
        r_w_t_correct++;
      }
      r_w_t_total++;
    }
  }
  double p = (double)(p_correct) / (double)(p_total);
  double r = (double)(r_correct) / (double)(r_total);
  double f = 2.0 * ((p * r) / (p + r));
  cerr << "seg: p: " << p << "\tr: " << r << "\tf: " << f << endl;

  p = (double)(p_w_t_correct) / (double)(p_w_t_total);
  r = (double)(r_w_t_correct) / (double)(r_w_t_total);
  f = 2.0 * ((p * r) / (p + r));
  cerr << "tag: p: " << p << "\tr: " << r << "\tf: " << f << endl;
  return f; 

}

void test_only(SegmentalRNN<LSTMBuilder>& segrnn,
          vector<pair<vector<int>,vector<pair<int,int>>>>& test_set,
          int max_seg_len = 0)
{
  for (auto& sent : test_set) {
    ComputationGraph cg;
    vector<pair<int, int>> yz_pred;
    vector<Expression> xins = segrnn.ConstructInput(sent.first, cg);
    segrnn.ViterbiDecode(xins, sent.second, cg, yz_pred, max_seg_len);
    unsigned int i;
    for(i = 0; i < yz_pred.size()-1; ++i){
      auto pred = yz_pred[i];
      cout << segrnn.td.Convert(pred.first) << ":" << pred.second << " ";
    }
    if(i >= 0 && (i == yz_pred.size()-1)){
      auto pred = yz_pred[i];
      cout << segrnn.td.Convert(pred.first) << ":" << pred.second;
    }
    cout << endl;
  }
}

void read_file(string file_path,
                     cnn::Dict& d,
                     cnn::Dict& td,
                     vector<pair<vector<int>,vector<pair<int,int>>>>& read_set,
                     bool test_only = false)
{
  read_set.clear();
  string line;
  cerr << "Reading data from " << file_path << "...\n";
  {
    ifstream in(file_path);
    assert(in);
    while(getline(in, line)) {
      read_set.push_back(ParseTrainingInstance(line, d, td, test_only));
    }
  }
  cerr << "Reading data from " << file_path << " finished \n";
}

void save_models(string model_file_prefix,
                    cnn::Dict& d,
                    cnn::Dict& td,
                    Model& model){
  cerr << "saving models..." << endl;

  const string f_name = model_file_prefix + ".params";
  ofstream out(f_name);
  boost::archive::text_oarchive oa(out);
  oa << model;
  out.close();

  const string f_d_name = model_file_prefix + ".dict";
  ofstream out_d(f_d_name);
  boost::archive::text_oarchive oa_d(out_d);
  oa_d << d;
  out_d.close();

  const string f_td_name = model_file_prefix + ".tdict";
  ofstream out_td(f_td_name);
  boost::archive::text_oarchive oa_td(out_td);
  oa_td << td;
  out_td.close();
  cerr << "saving models finished" << endl;
}

void load_models(string model_file_prefix,
                 Model& model){
  cerr << "loading models..." << endl;

  string fname = model_file_prefix + ".params";
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  in.close();

  cerr << "loading models finished" << endl;
}

void load_dicts(string model_file_prefix,
                 cnn::Dict& d,
                 cnn::Dict& td)
{
  cerr << "loading dicts..." << endl;
  string f_d_name = model_file_prefix + ".dict";
  ifstream in_d(f_d_name);
  boost::archive::text_iarchive ia_d(in_d);
  ia_d >> d;
  in_d.close();

  string f_td_name = model_file_prefix + ".tdict";
  ifstream in_td(f_td_name);
  boost::archive::text_iarchive ia_td(in_td);
  ia_td >> td;
  in_td.close();
  cerr << "loading dicts finished" << endl;
}

unsigned int edit_distance(const std::string& s1, const std::string& s2)
{
  const std::size_t len1 = s1.size(), len2 = s2.size();
  std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

  d[0][0] = 0;
  for(unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
    for(unsigned int i = 1; i <= len2; ++i) d[0][i] = i;
      for(unsigned int i = 1; i <= len1; ++i)
        for(unsigned int j = 1; j <= len2; ++j)
          d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) });
        return d[len1][len2];
}

double evaluate_partial(vector<vector<pair<int,int>>>& yz_preds,
                       vector<vector<pair<int,int>>>& yz_golds,
                       cnn::Dict& d,
                       cnn::Dict& td){
  assert(yz_preds.size() == yz_golds.size());

  int total_length_gold = 0;
  int total_length_pred = 0;
  int total_editing_distance = 0;

  for (int i = 0; i < yz_preds.size(); i++){
    // for sentence i
    vector<pair<int,int>>& yz_pred = yz_preds[i];
    vector<pair<int,int>>& yz_gold = yz_golds[i];
    std::string pred_s = "";
    for (auto e : yz_pred){
      pred_s.append(td.Convert(e.first));
      total_length_pred++;
    } 
    std::string gold_s = "";
    for (auto e : yz_gold){
      gold_s.append(td.Convert(e.first));
      total_length_gold++;
    }
    auto dis = edit_distance(pred_s, gold_s);
    total_editing_distance += dis;
    
  }
  cerr << "total_editing_distance: " << total_editing_distance << "\ttotal_length_gold: " << total_length_gold << "\ttotal_length_pred: " << total_length_pred << endl;
  double score = (((float) total_editing_distance) * 2.0)/((float)(total_length_gold + total_length_pred));
  cerr << "score: " << score << endl;
  return score; 
}

double predict_and_evaluate(SegmentalRNN<LSTMBuilder>& segrnn,
                            const vector<pair<vector<int>,vector<pair<int,int>>>>& input_set,
                            bool partial,
                            int max_seg_len,
                            string set_name = "DEV"
                            ){
  vector<vector<pair<int,int>>> yz_preds;
  vector<vector<pair<int,int>>> yz_golds;
  for (auto& sent : input_set) {
    ComputationGraph cg;
    vector<pair<int, int>> yz_pred;
    vector<Expression> xins = segrnn.ConstructInput(sent.first, cg);
    segrnn.ViterbiDecode(xins, sent.second, cg, yz_pred, max_seg_len);
    yz_golds.push_back(sent.second);
    yz_preds.push_back(yz_pred);
  }
  
  double f1 = evaluate(yz_preds, yz_golds, segrnn.d, segrnn.td);
  double f2 = evaluate_partial(yz_preds, yz_golds, segrnn.d, segrnn.td);

  double f = partial ? f2 : f1;
  cerr << set_name << endl;
  return f;
}


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  int test_max_seg_len;
  int max_consider_sentence_len;
  unsigned dev_every_i_reports;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("train", po::bool_switch()->default_value(false), "the training mode")
      ("hinge", po::bool_switch()->default_value(false), "training use the hinge loss")
      ("partial", po::bool_switch()->default_value(false), "the partially supervised training mode, where we don't have z")
      ("train_max_seg_len", po::value<int>(), "in the partially supervised training mode, the max len we consider")
      ("load_original_model", po::bool_switch()->default_value(false), "continuing the training by loading the model, only valid during training")
      ("max_consider_sentence_len", po::value<int>(&max_consider_sentence_len)->default_value(1000))
      ("dropout_rate", po::value<float>(), "dropout rate, also indicts using dropout during training")
      ("evaluate_test", po::bool_switch()->default_value(false), "evaluate test set every training iteration")
      ("test", po::bool_switch()->default_value(false), "the test mode")
      ("test_max_seg_len", po::value<int>(&test_max_seg_len)->default_value(30))
      ("dev_every_i_reports", po::value<unsigned>(&dev_every_i_reports)->default_value(1000))
      ("train_file", po::value<string>(), "path of the train file")
      ("dev_file", po::value<string>(), "path of the dev file")
      ("test_file", po::value<string>(), "path of the test file")
      ("model_file_prefix", po::value<string>(), "prefix path of the model files (and dictionaries)")
      ("upe", po::value<string>(), "use pre-trained word embeding")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
      cerr << desc << "\n";
      return 1;
  }

  if(vm["train"].as<bool>()){
    if (vm.count("upe")){
      cerr << "using pre-trained embeding from " << vm["upe"].as<string>() << endl;
      use_pretrained_embeding = true;
      pretrained_embeding = vm["upe"].as<string>();
    }else{
      use_pretrained_embeding = false;
      cerr << "not using pre-trained embeding" << endl;
    }

    if (vm.count("dropout_rate")){
      use_dropout = true;
      dropout_rate = vm["dropout_rate"].as<float>();
      cerr << "using dropout training, dropout rate: " << dropout_rate << endl;
    }else{
      use_dropout = false;
    }

    // create two dictionaries
    cnn::Dict d;
    cnn::Dict td;
    vector<pair<vector<int>,vector<pair<int,int>>>> training, dev, test;
    read_file(vm["train_file"].as<string>(), d, td, training);

    d.Freeze();  // no new word types allowed
    td.Freeze(); // no new tag types allowed
    d.SetUnk("<UNK>"); // set UNK to allow the unseen character in the dev and test set

    read_file(vm["dev_file"].as<string>(), d, td, dev);
    if (vm["evaluate_test"].as<bool>()){
      read_file(vm["test_file"].as<string>(), d, td, test);
    }

    Model model;
    // auto sgd = new SimpleSGDTrainer(&model);
    auto sgd = new AdamTrainer(&model, 1e-6, 0.0005, 0.01, 0.9999, 1e-8);
    int max_seg_len = DATA_MAX_SEG_LEN + 1;
    if(vm.count("train_max_seg_len")){
      max_seg_len = vm["train_max_seg_len"].as<int>();
    }

    cerr << "set max_seg_len = " << max_seg_len << endl;

    SegmentalRNN<LSTMBuilder> segrnn(model, d, td);

    if(vm["load_original_model"].as<bool>()){
      load_models(vm["model_file_prefix"].as<string>(), model);
    }

    double f_best = 0;
    unsigned report_every_i = 10;

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
        if(check_max_seg(sent.second, max_seg_len)){
          if (sent.first.size() > max_consider_sentence_len){
            cerr << "skip a sentence because its length " << sent.first.size() << " > " << max_consider_sentence_len << endl;
            continue;
          }
          vector<Expression> xins = segrnn.ConstructInput(sent.first, cg);
          if(vm["partial"].as<bool>()){
              if(vm["hinge"].as<bool>()){
                cerr << "Hingle Loss for partially supervised setting is not avaiable at this moment." << endl;
                abort();
              }else{
                segrnn.PartiallySupervisedCRFLoss(xins, sent.second, cg, max_seg_len);
              }
          }else{
            if(vm["hinge"].as<bool>()){
              segrnn.SupervisedHingeLoss(xins, sent.second, cg, max_seg_len);
            }else{
              segrnn.SupervisedCRFLoss(xins, sent.second, cg, max_seg_len);
            }
          }
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
        double f = predict_and_evaluate(segrnn, dev, vm["partial"].as<bool>(), max_seg_len);
        if (f > f_best) {
          f_best = f;
          save_models(vm["model_file_prefix"].as<string>(), d, td, model);
        }
        if (vm["evaluate_test"].as<bool>()){
          predict_and_evaluate(segrnn, test, vm["partial"].as<bool>(), max_seg_len, "TEST");
        }
      }
    }
    delete sgd;
  }else if(vm["test"].as<bool>()){
    use_pretrained_embeding = false;
    use_dropout = false;
    Model model;
    cnn::Dict d;
    cnn::Dict td;
    load_dicts(vm["model_file_prefix"].as<string>(), d, td);
    SegmentalRNN<LSTMBuilder> segrnn(model, d, td);
    load_models(vm["model_file_prefix"].as<string>(), model);
    vector<pair<vector<int>,vector<pair<int,int>>>> test;
    read_file(vm["test_file"].as<string>(), d, td, test, true);
    
    test_only(segrnn, test);
  }

}


