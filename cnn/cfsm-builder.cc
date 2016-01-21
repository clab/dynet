#include "cnn/cfsm-builder.h"

#include <fstream>
#include <iostream>

using namespace std;

namespace cnn {

using namespace expr;

inline bool is_ws(char x) { return (x == ' ' || x == '\t'); }
inline bool not_ws(char x) { return (x != ' ' && x != '\t'); }

NonFactoredSoftmaxBuilder::NonFactoredSoftmaxBuilder(unsigned rep_dim, unsigned vocab_size, Model* model) {
  p_w = model->add_parameters({vocab_size, rep_dim});
  p_b = model->add_parameters({vocab_size});
}

void NonFactoredSoftmaxBuilder::new_graph(ComputationGraph& cg) {
  pcg = &cg;
  w = parameter(cg, p_w);
  b = parameter(cg, p_b);
}

Expression NonFactoredSoftmaxBuilder::neg_log_softmax(const Expression& rep, unsigned wordidx) {
  return pickneglogsoftmax(affine_transform({b, w, rep}), wordidx);
}

unsigned NonFactoredSoftmaxBuilder::sample(const expr::Expression& rep) {
  softmax(affine_transform({b, w, rep}));
  vector<float> dist = as_vector(pcg->incremental_forward());
  unsigned c = 0;
  double p = rand01();
  for (; c < dist.size(); ++c) {
    p -= dist[c];
    if (p < 0.0) { break; }
  }
  if (c == dist.size()) {
    --c;
  }
  return c;
}

ClassFactoredSoftmaxBuilder::ClassFactoredSoftmaxBuilder(unsigned rep_dim,
                             const std::string& cluster_file,
                             Dict* word_dict,
                             Model* model) {
  ReadClusterFile(cluster_file, word_dict);
  const unsigned num_clusters = cdict.size();
  p_r2c = model->add_parameters({num_clusters, rep_dim});
  p_cbias = model->add_parameters({num_clusters});
  p_rc2ws.resize(num_clusters);
  p_rcwbiases.resize(num_clusters);
  for (unsigned i = 0; i < num_clusters; ++i) {
    auto& words = cidx2words[i];  // vector of word ids
    const unsigned num_words_in_cluster = words.size();
    if (num_words_in_cluster > 1) {
      // for singleton clusters, we don't need these parameters, so
      // we don't create them
      p_rc2ws[i] = model->add_parameters({num_words_in_cluster, rep_dim});
      p_rcwbiases[i] = model->add_parameters({num_words_in_cluster});
    }
  }
}

void ClassFactoredSoftmaxBuilder::new_graph(ComputationGraph& cg) {
  pcg = &cg;
  const unsigned num_clusters = cdict.size();
  r2c = parameter(cg, p_r2c);
  cbias = parameter(cg, p_cbias);
  rc2ws.clear();
  rc2biases.clear();
  rc2ws.resize(num_clusters);
  rc2biases.resize(num_clusters);
}

Expression ClassFactoredSoftmaxBuilder::neg_log_softmax(const Expression& rep, unsigned wordidx) {
  // TODO assert that new_graph has been called
  int clusteridx = widx2cidx[wordidx];
  assert(clusteridx >= 0);  // if this fails, wordid is missing from clusters
  Expression cscores = affine_transform({cbias, r2c, rep});
  Expression cnlp = pickneglogsoftmax(cscores, clusteridx);
  if (singleton_cluster[clusteridx]) return cnlp;
  // if there is only one word in the cluster, just return -log p(class | rep)
  // otherwise predict word too
  unsigned wordrow = widx2cwidx[wordidx];
  Expression& cwbias = get_rc2wbias(clusteridx);
  Expression& r2cw = get_rc2w(clusteridx);
  Expression wscores = affine_transform({cwbias, r2cw, rep});
  Expression wnlp = pickneglogsoftmax(wscores, wordrow);
  return cnlp + wnlp;
}

unsigned ClassFactoredSoftmaxBuilder::sample(const expr::Expression& rep) {
  // TODO assert that new_graph has been called
  Expression cscores = affine_transform({cbias, r2c, rep});
  softmax(cscores);
  auto cdist = as_vector(pcg->incremental_forward());
  unsigned c = 0;
  double p = rand01();
  for (; c < cdist.size(); ++c) {
    p -= cdist[c];
    if (p < 0.0) { break; }
  }
  if (c == cdist.size()) --c;
  unsigned w = 0;
  if (!singleton_cluster[c]) {
    Expression& cwbias = get_rc2wbias(c);
    Expression& r2cw = get_rc2w(c);
    Expression wscores = affine_transform({cwbias, r2cw, rep});
    softmax(wscores);
    auto wdist = as_vector(pcg->incremental_forward());
    p = rand01();
    for (; w < wdist.size(); ++w) {
      p -= wdist[w];
      if (p < 0.0) { break; }
    }
    if (w == wdist.size()) --w;
  }
  return cidx2words[c][w];
}

void ClassFactoredSoftmaxBuilder::ReadClusterFile(const std::string& cluster_file, Dict* word_dict) {
  cerr << "Reading clusters from " << cluster_file << " ...\n";
  ifstream in(cluster_file);
  assert(in);
  int wc = 0;
  string line;
  while(getline(in, line)) {
    ++wc;
    const unsigned len = line.size();
    unsigned startc = 0;
    while (is_ws(line[startc]) && startc < len) { ++startc; }
    unsigned endc = startc;
    while (not_ws(line[endc]) && endc < len) { ++endc; }
    unsigned startw = endc;
    while (is_ws(line[startw]) && startw < len) { ++startw; }
    unsigned endw = startw;
    while (not_ws(line[endw]) && endw < len) { ++endw; }
    assert(endc > startc);
    assert(startw > endc);
    assert(endw > startw);
    unsigned c = cdict.Convert(line.substr(startc, endc - startc));
    unsigned word = word_dict->Convert(line.substr(startw, endw - startw));
    if (word >= widx2cidx.size()) {
      widx2cidx.resize(word + 1, -1);
      widx2cwidx.resize(word + 1);
    }
    widx2cidx[word] = c;
    if (c >= cidx2words.size()) cidx2words.resize(c + 1);
    auto& clusterwords = cidx2words[c];
    widx2cwidx[word] = clusterwords.size();
    clusterwords.push_back(word);
  }
  singleton_cluster.resize(cidx2words.size());
  int scs = 0;
  for (unsigned i = 0; i < cidx2words.size(); ++i) {
    bool sc = cidx2words[i].size() <= 1;
    if (sc) scs++;
    singleton_cluster[i] = sc;
  }
  cerr << "Read " << wc << " words in " << cdict.size() << " clusters (" << scs << " singleton clusters)\n";
}

} // namespace cnn
