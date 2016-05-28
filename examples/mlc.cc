#include "cnn/timing.h"
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"
#include "cnn/grad-check.h"

#include <sstream>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

struct TrainingInstance {
  TrainingInstance() {}
  TrainingInstance(const vector<pair<unsigned,float>>& x, const vector<unsigned>& y) : feats(x), labels(y) {}
  vector<pair<unsigned,float>> feats;  // sparse representation of x vector
  vector<unsigned> labels;  // labels
};

void swap(TrainingInstance& a, TrainingInstance& b) {
  using std::swap;
  swap(a.feats, b.feats);
  swap(a.labels, b.labels);
}

// X: 0 864:0.0497399 1523:0.0446641 1681:0.0673872 2293:0.0718105 2845:0.0657134 2867:0.0653402 3240:0.0795168 4125:0.0423215 4271:0.0691369 4665:0.0500863 5216:0.252185 5573:0.0672562 5699:0.0594998 5794:0.0737821 6222:0.124501 6592:0.101431 7227:0.194091 7975:0.0766401
// Y: 0 35:1 60:1 94:1 95:1 103:1
vector<TrainingInstance> ReadFiles(const char* xfname, const char* yfname, unsigned& maxfeat, unsigned& maxlabel) {
  maxfeat = 0;
  maxlabel = 0;
  vector<TrainingInstance> d;
  ifstream inx(xfname);
  assert(inx);
  ifstream iny(yfname);
  assert(iny);
  string linex, liney;
  string tok;
  while(getline(inx,linex)) {
    getline(iny,liney);

    vector<pair<unsigned,float>> v;
    istringstream isx(linex);
    isx >> tok;
    assert(tok == "0");
    while (isx) {
      isx >> tok;
      if (!isx) break;
      size_t pos = tok.find(':');
      assert(pos != string::npos);
      tok[pos] = 0;
      unsigned fi = atoi(&tok[0]);
      if (fi > maxfeat) maxfeat = fi;
      float fv = strtof(&tok[pos+1], 0);
      v.emplace_back(make_pair(fi, fv));
    }
    vector<unsigned> y;
    istringstream isy(liney);
    isy >> tok;
    assert(tok == "0");
    while (isy) {
      isy >> tok;
      if (!isy) break;
      size_t pos = tok.find(':');
      assert(pos != string::npos);
      tok[pos] = 0;
      unsigned yi = atoi(&tok[0]);
      if (yi > maxlabel) maxlabel = yi;
      y.push_back(yi);
    }
    d.emplace_back(v, y);
  }
  return d;
}

struct MLCBuilder {
  explicit MLCBuilder(Model& m, unsigned nfeats, unsigned labels) {
    unsigned HIDDEN_SIZE = 200;
    p_xe = m.add_lookup_parameters(nfeats, {HIDDEN_SIZE});
    p_bh = m.add_parameters({HIDDEN_SIZE});
    p_h2y = m.add_parameters({labels, HIDDEN_SIZE});
    p_by = m.add_parameters({labels});
  }
  // output will be a vector of scores that can be 'softmaxed' or 'sparsemaxed'
  // into a probability distribution, or it can be compared with a target
  // distribution and a loss will be computed
  Expression BuildPredictionScores(ComputationGraph& cg, const vector<pair<unsigned,float>>& feats) const {
    vector<Expression> fe(feats.size() + 1);
    unsigned fi = 0;
    for (auto& xi : feats) {
      fe[fi++] = lookup(cg, p_xe, xi.first) * xi.second;  // xi.second is the input feature value
    }
    fe[fi] = parameter(cg, p_bh); // put bias term at the end
    Expression h = tanh(sum(fe));
    Expression h2y = parameter(cg, p_h2y);
    Expression by = parameter(cg, p_by);
    return affine_transform({by, h2y, h});
  }
  LookupParameters* p_xe;
  Parameters* p_bh;
  Parameters* p_h2y;
  Parameters* p_by;
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  if (argc != 5) {
    cerr << "Usage: " << argv[0] << " x.train.txt y.train.txt x.dev.txt y.dev.txt\n";
    return 1;
  }
  vector<TrainingInstance> train, dev;
  unsigned max_xi, max_yi, dxi, dyi;
  train = ReadFiles(argv[1], argv[2], max_xi, max_yi);
  cerr << "Maximum feature index: " << max_xi << endl;
  cerr << "Maximum label index: " << max_yi << endl;
  dev = ReadFiles(argv[3], argv[4], dxi, dyi);
  assert(dxi <= max_xi);
  assert(dyi <= max_yi);
  max_xi++;
  max_yi++;

  // parameters
  Model m;
  MLCBuilder mlc(m, max_xi, max_yi);

  //AdadeltaTrainer sgd(&m);
  SimpleSGDTrainer sgd(&m);
  sgd.eta0 = 0.001;
  sgd.eta = 0.001;

  unsigned report_every_i = 50;
  unsigned si = train.size();
  bool first = true;
  vector<unsigned> order(train.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  double ti = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned instances = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == train.size()) {
        si = 0;
        if (first) { first = false; } else { sgd.update_epoch(); }
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }
      // build graph for this instance
      ComputationGraph cg;
      auto& xy = train[order[si]];
      ++si;
      ++instances;
      ++ti;
      Expression u = mlc.BuildPredictionScores(cg, xy.feats);

      if (rand01() < 0.004) {
        sparsemax(u * 1.5);  // this increases sparsity at test time, which Andre found the be useful
        vector<float> p = as_vector(cg.incremental_forward());
        for (unsigned j = 0; j < p.size(); ++j)
          if (p[j] > 0) cerr << j << ' ';
        cerr << " |||";
        for (auto y : xy.labels)
          cerr << ' ' << y;
        cerr << endl;
      }
      sparsemax_loss(u, &xy.labels);
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd.update(1.0);
    }
    cerr << "[epoch=" << (ti / train.size()) << "] E=" << (loss / instances) << ' ';
  }
}

