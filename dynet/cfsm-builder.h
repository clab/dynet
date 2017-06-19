#ifndef DYNET_CFSMBUILDER_H
#define DYNET_CFSMBUILDER_H

#include <vector>
#include <string>

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/dict.h"

namespace dynet {

class SoftmaxBuilder {
public:
  virtual ~SoftmaxBuilder();

  // call this once per ComputationGraph
  virtual void new_graph(ComputationGraph& cg) = 0;

  // -log(p(w | rep))
  virtual Expression neg_log_softmax(const Expression& rep, unsigned wordidx) = 0;

  // samples a word from p(w | rep)
  virtual unsigned sample(const Expression& rep) = 0;

  // returns an Expression representing a vector the size of the vocabulary.
  // The ith dimension gives log p(w_i | rep). This function may be SLOW. Avoid if possible.
  virtual Expression full_log_distribution(const Expression& rep) = 0;
};

class StandardSoftmaxBuilder : public SoftmaxBuilder {
public:
  StandardSoftmaxBuilder(unsigned rep_dim, unsigned vocab_size, ParameterCollection& model);
  void new_graph(ComputationGraph& cg);
  Expression neg_log_softmax(const Expression& rep, unsigned wordidx);
  unsigned sample(const Expression& rep);
  Expression full_log_distribution(const Expression& rep);
  ParameterCollection & get_parameter_collection() { return local_model; }
private:
  StandardSoftmaxBuilder();
  Parameter p_w;
  Parameter p_b;
  Expression w;
  Expression b;
  ComputationGraph* pcg;
  ParameterCollection local_model;
};

// helps with implementation of hierarchical softmax
// read a file with lines of the following format
// CLASSID   word    [freq]
class ClassFactoredSoftmaxBuilder : public SoftmaxBuilder {
 public:
  ClassFactoredSoftmaxBuilder(unsigned rep_dim,
                              const std::string& cluster_file,
                              Dict& word_dict,
                              ParameterCollection& model);

  void new_graph(ComputationGraph& cg);
  Expression neg_log_softmax(const Expression& rep, unsigned wordidx);
  unsigned sample(const Expression& rep);
  Expression full_log_distribution(const Expression& rep);
  void initialize_expressions();

  ParameterCollection & get_parameter_collection() { return local_model; }

 private:
  ClassFactoredSoftmaxBuilder();
  void read_cluster_file(const std::string& cluster_file, Dict& word_dict);

  Dict cdict;
  std::vector<int> widx2cidx; // will be -1 if not present
  std::vector<unsigned> widx2cwidx; // word index to word index inside of cluster
  std::vector<std::vector<unsigned>> cidx2words;
  std::vector<bool> singleton_cluster; // does cluster contain a single word type?

  ParameterCollection local_model;
  // parameters
  Parameter p_r2c;
  Parameter p_cbias;
  std::vector<Parameter> p_rc2ws;     // len = number of classes
  std::vector<Parameter> p_rcwbiases; // len = number of classes

  // Expressions for current graph
  inline Expression& get_rc2w(unsigned cluster_idx) {
    Expression& e = rc2ws[cluster_idx];
    if (!e.pg)
      e = parameter(*pcg, p_rc2ws[cluster_idx]);
    return e;
  }
  inline Expression& get_rc2wbias(unsigned cluster_idx) {
    Expression& e = rc2biases[cluster_idx];
    if (!e.pg)
      e = parameter(*pcg, p_rcwbiases[cluster_idx]);
    return e;
  }
  ComputationGraph* pcg;
  Expression r2c;
  Expression cbias;
  std::vector<Expression> rc2ws;
  std::vector<Expression> rc2biases;
};
}  // namespace dynet

#endif
