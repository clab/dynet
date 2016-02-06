#ifndef CNN_HSMBUILDER_H
#define CNN_HSMBUILDER_H

#include <vector>
#include <string>
#include <unordered_map>
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/dict.h"
#include "cnn/cfsm-builder.h"

namespace cnn {

struct Parameters;

class Cluster {
private:
  std::vector<Cluster*> children;
  std::vector<unsigned> path;
  std::vector<unsigned> terminals;
  std::unordered_map<unsigned, unsigned> word2ind;
  Parameters* p_weights;
  Parameters* p_bias;
  mutable expr::Expression weights;
  mutable expr::Expression bias;
  bool initialized;
  unsigned output_size;

  expr::Expression predict(expr::Expression h, ComputationGraph& cg) const;

public:
  Cluster();
  Cluster* add_child(unsigned sym);
  void add_word(unsigned word);
  void initialize(unsigned rep_dim, Model* model);

  void new_graph(ComputationGraph& cg);
  unsigned sample(expr::Expression h, ComputationGraph& cg) const;
  expr::Expression neg_log_softmax(expr::Expression h, unsigned r, ComputationGraph& cg) const;

  unsigned get_index(unsigned word) const;
  unsigned get_word(unsigned index) const;
  unsigned num_children() const;
  const Cluster* get_child(unsigned i) const;
  const std::vector<unsigned>& get_path() const;
  expr::Expression get_weights(ComputationGraph& cg) const;
  expr::Expression get_bias(ComputationGraph& cg) const;

  std::string toString() const;
};

// helps with implementation of hierarchical softmax
// read a file with lines of the following format
// CLASSID   word    [freq]
class HierarchicalSoftmaxBuilder : public FactoredSoftmaxBuilder {
 public:
  HierarchicalSoftmaxBuilder(unsigned rep_dim,
                              const std::string& cluster_file,
                              Dict* word_dict,
                              Model* model);
  ~HierarchicalSoftmaxBuilder();
  // call this once per ComputationGraph
  void new_graph(ComputationGraph& cg);

  // -log(p(c | rep) * p(w | c, rep))
  expr::Expression neg_log_softmax(const expr::Expression& rep, unsigned wordidx);

  // samples a word from p(w,c | rep)
  unsigned sample(const expr::Expression& rep);

 private:
  Cluster* ReadClusterFile(const std::string& cluster_file, Dict* word_dict);
  std::vector<Cluster*> widx2path; // will be NULL if not found
  Dict path_symbols;

  ComputationGraph* pcg;
  Cluster* root;
};

}  // namespace cnn

#endif
