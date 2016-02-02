#ifndef CNN_CFSMBUILDER_H
#define CNN_CFSMBUILDER_H

#include <vector>
#include <string>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/dict.h"

namespace cnn {

class SoftmaxBuilder {
public:
  virtual ~SoftmaxBuilder();

  // call this once per ComputationGraph
  virtual void new_graph(ComputationGraph& cg) = 0;

  // -log(p(c | rep) * p(w | c, rep))
  virtual expr::Expression neg_log_softmax(const expr::Expression& rep, unsigned wordidx) = 0;

  // samples a word from p(w,c | rep)
  virtual unsigned sample(const expr::Expression& rep) = 0;

  // add parameters to a model. Usually called after deserializing.
  virtual void initialize(Model& model) = 0;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class StandardSoftmaxBuilder : public SoftmaxBuilder {
public:
  StandardSoftmaxBuilder(unsigned rep_dim, unsigned vocab_size, Model* model);
  void new_graph(ComputationGraph& cg);
  expr::Expression neg_log_softmax(const expr::Expression& rep, unsigned wordidx);
  unsigned sample(const expr::Expression& rep);
  void initialize(Model& model);

private:
  StandardSoftmaxBuilder();
  unsigned rep_dim, vocab_size;
  ParameterIndex p_w;
  ParameterIndex p_b;
  expr::Expression w;
  expr::Expression b;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<StandardSoftmaxBuilder, SoftmaxBuilder>();
    std::cerr << "serializing standardsoftmaxbuilder" << std::endl;
    ar & rep_dim;
    ar & vocab_size;
  }
};

// helps with implementation of hierarchical softmax
// read a file with lines of the following format
// CLASSID   word    [freq]
class ClassFactoredSoftmaxBuilder : public SoftmaxBuilder {
 public:
  ClassFactoredSoftmaxBuilder(unsigned rep_dim,
                              const std::string& cluster_file,
                              Dict* word_dict,
                              Model* model);

  void new_graph(ComputationGraph& cg);
  expr::Expression neg_log_softmax(const expr::Expression& rep, unsigned wordidx);
  unsigned sample(const expr::Expression& rep);
  void initialize(Model& model);

 private:
  ClassFactoredSoftmaxBuilder();
  void ReadClusterFile(const std::string& cluster_file, Dict* word_dict);

  unsigned rep_dim;
  Dict cdict;
  std::vector<int> widx2cidx; // will be -1 if not present
  std::vector<unsigned> widx2cwidx; // word index to word index inside of cluster
  std::vector<std::vector<unsigned>> cidx2words;
  std::vector<bool> singleton_cluster; // does cluster contain a single word type?

  // parameters
  ParameterIndex p_r2c;
  ParameterIndex p_cbias;
  std::vector<ParameterIndex> p_rc2ws;     // len = number of classes
  std::vector<ParameterIndex> p_rcwbiases; // len = number of classes

  // Expressions for current graph
  inline expr::Expression& get_rc2w(unsigned cluster_idx) {
    expr::Expression& e = rc2ws[cluster_idx];
    if (!e.pg)
      e = expr::parameter(*pcg, p_rc2ws[cluster_idx]);
    return e;
  }
  inline expr::Expression& get_rc2wbias(unsigned cluster_idx) {
    expr::Expression& e = rc2biases[cluster_idx];
    if (!e.pg)
      e = expr::parameter(*pcg, p_rcwbiases[cluster_idx]);
    return e;
  }
  ComputationGraph* pcg;
  expr::Expression r2c;
  expr::Expression cbias;
  std::vector<expr::Expression> rc2ws;
  std::vector<expr::Expression> rc2biases;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<ClassFactoredSoftmaxBuilder, SoftmaxBuilder>();
    ar & rep_dim;
    ar & cdict;
    ar & widx2cidx;
    ar & widx2cwidx;
    ar & cidx2words;
    ar & singleton_cluster;
  }
};
}  // namespace cnn
BOOST_CLASS_EXPORT_KEY(cnn::StandardSoftmaxBuilder)
BOOST_CLASS_EXPORT_KEY(cnn::ClassFactoredSoftmaxBuilder)

#endif
