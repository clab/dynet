#ifndef DYNET_HSMBUILDER_H
#define DYNET_HSMBUILDER_H

#include <vector>
#include <string>
#include <unordered_map>
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/dict.h"
#include "dynet/cfsm-builder.h"

namespace dynet {

/**
 * @brief Cluster softmax
 * @details This is used in the hierarchical softmax
 * 
 */
class Cluster {
private:
  std::vector<Cluster*> children;
  std::vector<unsigned> path;
  std::vector<unsigned> terminals;
  std::unordered_map<unsigned, unsigned> word2ind;
  Parameter p_weights;
  Parameter p_bias;
  mutable Expression weights;
  mutable Expression bias;
  unsigned rep_dim;
  unsigned output_size;
  bool update;

  Expression predict(Expression h, ComputationGraph& cg) const;

public:
  Cluster();
  Cluster* add_child(unsigned sym);
  void add_word(unsigned word);
  void initialize(ParameterCollection& model);
  void initialize(unsigned rep_dim, ParameterCollection& model);

  void new_graph(ComputationGraph& cg, bool update=true);
  unsigned sample(Expression h, ComputationGraph& cg) const;
  Expression neg_log_softmax(Expression h, unsigned r, ComputationGraph& cg) const;

  unsigned get_index(unsigned word) const;
  unsigned get_word(unsigned index) const;
  unsigned num_children() const;
  const Cluster* get_child(unsigned i) const;
  const std::vector<unsigned>& get_path() const;
  Expression get_weights(ComputationGraph& cg) const;
  Expression get_bias(ComputationGraph& cg) const;

  std::string toString() const;
};

// helps with implementation of hierarchical softmax
// read a file with lines of the following format
// CLASSID   word    [freq]
class HierarchicalSoftmaxBuilder : public SoftmaxBuilder {
 public:
  HierarchicalSoftmaxBuilder(unsigned rep_dim,
                              const std::string& cluster_file,
                              Dict& word_dict,
                              ParameterCollection& model);
  ~HierarchicalSoftmaxBuilder();

  void initialize(ParameterCollection& model);

  // call this once per ComputationGraph
  void new_graph(ComputationGraph& cg, bool update=true);

  // -log(p(c | rep) * p(w | c, rep))
  Expression neg_log_softmax(const Expression& rep, unsigned wordidx);
  Expression neg_log_softmax(const Expression& rep, const std::vector<unsigned>& classidxs){return Expression();}

  // samples a word from p(w,c | rep)
  unsigned sample(const Expression& rep);

  Expression full_log_distribution(const Expression& rep);
  Expression full_logits(const Expression& rep);
  
  ParameterCollection & get_parameter_collection() { return local_model; }

 private:
  ParameterCollection local_model;
  Cluster* read_cluster_file(const std::string& cluster_file, Dict& word_dict);
  std::vector<Cluster*> widx2path; // will be NULL if not found
  Dict path_symbols;

  ComputationGraph* pcg;
  Cluster* root;
};
}  // namespace dynet

#endif
