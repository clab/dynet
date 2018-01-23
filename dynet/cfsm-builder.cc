#include "dynet/cfsm-builder.h"
#include "dynet/except.h"
#include "dynet/param-init.h"

#include <fstream>
#include <iostream>

using namespace std;

namespace dynet {

inline bool is_ws(char x) { return (x == ' ' || x == '\t'); }
inline bool not_ws(char x) { return (x != ' ' && x != '\t'); }

SoftmaxBuilder::~SoftmaxBuilder() {}

StandardSoftmaxBuilder::StandardSoftmaxBuilder() {}

StandardSoftmaxBuilder::StandardSoftmaxBuilder(unsigned rep_dim, unsigned vocab_size, ParameterCollection& model, bool bias) : bias(bias) {
  local_model = model.add_subcollection("standard-softmax-builder");
  p_w = local_model.add_parameters({vocab_size, rep_dim});
  if (bias)
    p_b = local_model.add_parameters({vocab_size}, ParameterInitConst(0.f));
}
StandardSoftmaxBuilder::StandardSoftmaxBuilder(Parameter& p_w, Parameter& p_b) : bias(true){
  this->p_w = p_w;
  this->p_b = p_b;
  this->local_model = *(this->p_w.get_storage().owner);
}

StandardSoftmaxBuilder::StandardSoftmaxBuilder(Parameter& p_w) : bias(false){
  this->p_w = p_w;
  this->local_model = *(this->p_w.get_storage().owner);
}


void StandardSoftmaxBuilder::new_graph(ComputationGraph& cg, bool update) {
  pcg = &cg;
  w = update ? parameter(cg, p_w) : const_parameter(cg, p_w);
  if (bias)
    b = update ? parameter(cg, p_b) : const_parameter(cg, p_b);
}

Expression StandardSoftmaxBuilder::neg_log_softmax(const Expression& rep, unsigned classidx) {
  return pickneglogsoftmax(full_logits(rep), classidx);
}

Expression StandardSoftmaxBuilder::neg_log_softmax(const Expression& rep, const std::vector<unsigned>& classidxs) {
  DYNET_ARG_CHECK(rep.dim().bd == classidxs.size(), "Inputs of StandardSoftmaxBuilder::neg_log_softmax should have same batch size, got " << rep.dim().bd << " for rep and " << classidxs.size() << " for classidxs");
  return pickneglogsoftmax(full_logits(rep), classidxs);
}

unsigned StandardSoftmaxBuilder::sample(const Expression& rep) {
  Expression dist_expr = bias ? softmax(affine_transform({b, w, rep})) : w * rep;
  vector<float> dist = as_vector(pcg->incremental_forward(dist_expr));
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

Expression StandardSoftmaxBuilder::full_log_distribution(const Expression& rep) {
  return log_softmax(full_logits(rep));
}

Expression StandardSoftmaxBuilder::full_logits(const Expression& rep) {
  if (bias)
    return affine_transform({b, w, rep});
  else
    return w * rep;
}

ClassFactoredSoftmaxBuilder::ClassFactoredSoftmaxBuilder() {}

ClassFactoredSoftmaxBuilder::ClassFactoredSoftmaxBuilder(unsigned rep_dim,
                             const std::string& cluster_file,
                             Dict& word_dict,
                             ParameterCollection& model,
                             bool bias) : bias(bias){
  read_cluster_file(cluster_file, word_dict);
  const unsigned num_clusters = cdict.size();
  local_model = model.add_subcollection("class-factored-softmax-builder");
  p_r2c = local_model.add_parameters({num_clusters, rep_dim});
  if (bias)
    p_cbias = local_model.add_parameters({num_clusters}, ParameterInitConst(0.f));
  p_rc2ws.resize(num_clusters);
  if (bias)
    p_rcwbiases.resize(num_clusters);
  for (unsigned i = 0; i < num_clusters; ++i) {
    auto& words = cidx2words[i];  // vector of word ids
    const unsigned num_words_in_cluster = words.size();
    if (num_words_in_cluster > 1) {
      // for singleton clusters, we don't need these parameters, so
      // we don't create them
      p_rc2ws[i] = local_model.add_parameters({num_words_in_cluster, rep_dim});
      if (bias)
        p_rcwbiases[i] = local_model.add_parameters({num_words_in_cluster}, ParameterInitConst(0.f));
    }
  }
}

void ClassFactoredSoftmaxBuilder::new_graph(ComputationGraph& cg, bool update) {
  pcg = &cg;
  const unsigned num_clusters = cdict.size();
  r2c = update ? parameter(cg, p_r2c) : const_parameter(cg, p_r2c);
  if (bias)
    cbias = update ? parameter(cg, p_cbias) : const_parameter(cg, p_cbias);
  rc2ws.clear();
  rc2biases.clear();
  rc2ws.resize(num_clusters);
  rc2biases.resize(num_clusters);
  this->update = update;
}

Expression ClassFactoredSoftmaxBuilder::neg_log_softmax(const Expression& rep, unsigned classidx) {
  // TODO check that new_graph has been called
  int clusteridx = widx2cidx[classidx];
  DYNET_ARG_CHECK(clusteridx >= 0,
                          "Word ID " << classidx << " missing from clusters in ClassFactoredSoftmaxBuilder::neg_log_softmax");
  Expression cscores = class_logits(rep);
  Expression cnlp = pickneglogsoftmax(cscores, clusteridx);
  if (singleton_cluster[clusteridx]) return cnlp;
  // if there is only one word in the cluster, just return -log p(class | rep)
  // otherwise predict word too
  unsigned wordrow = widx2cwidx[classidx];
  // Expression& cwbias = get_rc2wbias(clusteridx);
  Expression& r2cw = get_rc2w(clusteridx);
  // Expression wscores = affine_transform({cwbias, r2cw, rep});
  Expression wscores = (bias) ? affine_transform({get_rc2wbias(clusteridx), r2cw, rep}) : (r2cw * rep);
  Expression wnlp = pickneglogsoftmax(wscores, wordrow);
  return cnlp + wnlp;
}

Expression ClassFactoredSoftmaxBuilder::neg_log_softmax(const Expression& rep, const std::vector<unsigned>& classidxs) {
  unsigned batch_size = classidxs.size();
  std::vector<Expression> nlps;
  for (unsigned i=0;i<batch_size;++i)
    nlps.push_back(neg_log_softmax(pick_batch_elem(rep, i), classidxs[i]));
  
  return concatenate_to_batch(nlps);
}

unsigned ClassFactoredSoftmaxBuilder::sample(const Expression& rep) {
  // TODO check that new_graph has been called
  Expression cscores = class_logits(rep);
  Expression cdist_expr = softmax(cscores);
  auto cdist = as_vector(pcg->incremental_forward(cdist_expr));
  unsigned c = 0;
  double p = rand01();
  for (; c < cdist.size(); ++c) {
    p -= cdist[c];
    if (p < 0.0) { break; }
  }
  if (c == cdist.size()) --c;
  unsigned w = 0;
  if (!singleton_cluster[c]) {
    Expression wscores = subclass_logits(rep, c);
    Expression wdist_expr = softmax(wscores);
    auto wdist = as_vector(pcg->incremental_forward(wdist_expr));
    p = rand01();
    for (; w < wdist.size(); ++w) {
      p -= wdist[w];
      if (p < 0.0) { break; }
    }
    if (w == wdist.size()) --w;
  }
  return cidx2words[c][w];
}

Expression ClassFactoredSoftmaxBuilder::full_log_distribution(const Expression& rep) {
  return log_softmax(full_logits(rep));
}

Expression ClassFactoredSoftmaxBuilder::full_logits(const Expression& rep) {
  vector<Expression> full_dist(widx2cidx.size());
  Expression cscores = class_log_distribution(rep);
  for (unsigned i = 0; i < widx2cidx.size(); ++i) {
    if (widx2cidx[i] == -1) {
      // XXX: Should be -inf
      full_dist[i] = input(*pcg, -10000);
    }
  }

  for (unsigned c = 0; c < p_rc2ws.size(); ++c) {
    Expression cscore = pick(cscores, c);
    if (singleton_cluster[c]) {
      for (unsigned i = 0; i < cidx2words[c].size(); ++i) {
        unsigned w = cidx2words[c][i];
        full_dist[w] = cscore;
      }
    }
    else {
      Expression wscores = subclass_logits(rep, c);
      Expression wdist = softmax(wscores);

      for (unsigned i = 0; i < cidx2words[c].size(); ++i) {
        unsigned w = cidx2words[c][i];
        full_dist[w] = pick(wdist, i) + cscore;
      }
    }
  }

  return concatenate(full_dist);
}

Expression ClassFactoredSoftmaxBuilder::class_log_distribution(const Expression& rep) {
  return log_softmax(class_logits(rep));
}

Expression ClassFactoredSoftmaxBuilder::class_logits(const Expression& rep) {
  if (bias)
    return affine_transform({cbias, r2c, rep});
  else
    return r2c * rep;
}

Expression ClassFactoredSoftmaxBuilder::subclass_log_distribution(const Expression& rep, unsigned clusteridx) {
  return log_softmax(subclass_logits(rep, clusteridx));
}

Expression ClassFactoredSoftmaxBuilder::subclass_logits(const Expression& rep, unsigned clusteridx) {
  Expression& r2cw = get_rc2w(clusteridx);
  if (bias){
    Expression& cwbias = get_rc2wbias(clusteridx);
    return affine_transform({cwbias, r2cw, rep});
  } else
    return r2cw * rep;
}

void ClassFactoredSoftmaxBuilder::read_cluster_file(const std::string& cluster_file, Dict& word_dict) {
  cerr << "Reading clusters from " << cluster_file << " ...\n";
  ifstream in(cluster_file);
  if(!in)
    DYNET_INVALID_ARG("Could not find cluster file " << cluster_file << " in ClassFactoredSoftmax");
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
    if(endc <= startc || startw <= endc || endw <= startw)
      DYNET_INVALID_ARG("Invalid format in cluster file " << cluster_file << " in ClassFactoredSoftmax");
    unsigned c = cdict.convert(line.substr(startc, endc - startc));
    unsigned word = word_dict.convert(line.substr(startw, endw - startw));
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

void ClassFactoredSoftmaxBuilder::initialize_expressions() {
  for (unsigned c = 0; c < p_rc2ws.size(); ++c) {
    //get_rc2w(_bias) creates the expression at c if the expression does not already exist.
    get_rc2w(c);
    get_rc2wbias(c);
  }
}

} // namespace dynet
