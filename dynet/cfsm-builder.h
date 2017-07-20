/**
 * \file csfm-builder.h
 * \defgroup softmaxbuilders softmaxbuilders
 * \brief Helper structures to build softmax layers
 *
 * \details TODO: Create documentation and explain rnns, etc...
 */

#ifndef DYNET_CFSMBUILDER_H
#define DYNET_CFSMBUILDER_H

#include <vector>
#include <string>

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/dict.h"

namespace dynet {

/**
 * \ingroup softmaxbuilders
 * \brief Interface for building softmax layers
 * \details A softmax layer returns a probability distribution over \f$C\f$ classes given a vector \f$h\in\mathbb R^d\f$, with 
 * 
 * \f$p(c)\propto \exp(W_i^Th + b_i)\ \forall i\in\{1\ldots C\}\f$
 * 
 * Where \f$W\in \mathbb R^{C\times d}, b \in \mathbb R^C\f$
 */
class SoftmaxBuilder {
public:
  virtual ~SoftmaxBuilder();

  /**
   * \brief This initializes the parameters in the computation graph
   * \details Call this once per ComputationGraph before any computation with the softmax
   * 
   * \param cg Computation graph
   * \param update Whether to update the parameters
   */
  virtual void new_graph(ComputationGraph& cg, bool update=true) = 0;

  /**
   * \brief Negative log probability of a class
   * \details Given class \f$c\f$ and vector \f$h\f$, this returns \f$-\log(p(c \mid h))\f$
   * 
   * \param rep vector expression
   * \param class Class
   * 
   * \return \f$-\log(p(\texttt{class} \mid \texttt{rep}))\f$
   */
  virtual Expression neg_log_softmax(const Expression& rep, unsigned classidx) = 0;

    /**
   * \brief Batched version of the former
   * \details Returns a batched scalar
   * 
   * \param rep Vector expression (batched)
   * \param classes List of classes, one per batch element
   * 
   * \return \f$-\log(p(\texttt{class}_b \mid \texttt{rep}_b))\f$ for each batch element \f$b\f$
   */
  virtual Expression neg_log_softmax(const Expression& rep, const std::vector<unsigned>& classidxs) = 0;

  /**
   * \brief Sample from the softmax distribution
   * 
   * \param rep Vector expression parametrizing the distribution
   * \return Sampled class
   */
  virtual unsigned sample(const Expression& rep) = 0;

  /**
   * \brief Returns an Expression representing a vector the size of the number of classes.
   * \details The ith dimension gives \f$\log p(c_i | \texttt{rep})\f$. This function may be SLOW. Avoid if possible.
   * 
   * \param rep Vector expression parametrizing the distribution
   * \return Expression of the distribution
   */
  virtual Expression full_log_distribution(const Expression& rep) = 0;
  /**
   * \brief Returns the logits (before application of the softmax)
   * \details The ith dimension gives \f$W_i^Th + b_i\f$
   * 
   * \param rep Vector expression parametrizing the distribution
   * \return Expression for the logits
   */
  virtual Expression full_logits(const Expression& rep) = 0;

  /**
   * \brief Returns the ParameterCollection containing the softmax parameters
   * 
   * \return ParameterCollection
   */
  virtual ParameterCollection & get_parameter_collection() = 0;
};

/**
 * \ingroup softmaxbuilders
 * \brief This class implements the standard Softmax
 * 
 */
class StandardSoftmaxBuilder : public SoftmaxBuilder {
public:
  /**
   * \brief Constructs a softmaxbuilder
   * \details This creates the parameters given the dimensions
   * 
   * \param rep_dim Dimension of the input vectors
   * \param num_classes Number of classes
   * \param pc Parameter collection
   * \param bias Whether to use a bias vector or not
   */
  StandardSoftmaxBuilder(unsigned rep_dim, unsigned num_classes, ParameterCollection& pc,bool bias=true);
  /**
   * \brief Builds a softmax layer with pre-existing parameters
   * 
   * \param p_w Weight matrix
   * \param p_b Bias vector
   */
  StandardSoftmaxBuilder(Parameter& p_w, Parameter& p_b);
  /**
   * \brief Builds a softmax layer with pre-existing parameters (no bias)
   * 
   * \param p_w Weight matrix
   */
  StandardSoftmaxBuilder(Parameter& p_w);
  void new_graph(ComputationGraph& cg, bool update=true);
  Expression neg_log_softmax(const Expression& rep, unsigned classidx);
  Expression neg_log_softmax(const Expression& rep, const std::vector<unsigned>& classidxs);
  unsigned sample(const Expression& rep);
  Expression full_log_distribution(const Expression& rep);
  Expression full_logits(const Expression& rep);
  ParameterCollection & get_parameter_collection() { return local_model; }
private:
  StandardSoftmaxBuilder();
  Parameter p_w;
  Parameter p_b;
  Expression w;
  Expression b;
  ComputationGraph* pcg;
  ParameterCollection local_model;
  bool bias;
};

// helps with implementation of hierarchical softmax
/**
 * \ingroup softmaxbuilders
 * \brief Class factored softmax
 * \details Each class is separated into a subclass, ie \f$p(i\mid h)=p(i\mid h, c) p(c\mid h)\f$ where \f$c\f$ is a class and \f$i\f$ a subclass
 * 
 */
class ClassFactoredSoftmaxBuilder : public SoftmaxBuilder {
 public:
  /**
   * \brief Constructor from file
   * \details This constructs the CFSM from a file with lines of the following format
   * 
   *     CLASSID   word    [freq]
   * 
   * For words for instance
   * 
   * \param rep_dim Dimension of the input vector
   * \param cluster_file File containing classes
   * \param word_dict Dictionary for words (maps words to index)
   * \param pc ParameterCollection
   * \param bias Whether to use a bias vector or not
   */
  ClassFactoredSoftmaxBuilder(unsigned rep_dim,
                              const std::string& cluster_file,
                              Dict& word_dict,
                              ParameterCollection& pc,
                              bool bias=true);

  void new_graph(ComputationGraph& cg, bool update=true);
  Expression neg_log_softmax(const Expression& rep, unsigned classidx);
  Expression neg_log_softmax(const Expression& rep, const std::vector<unsigned>& classidxs);
  unsigned sample(const Expression& rep);
  Expression full_log_distribution(const Expression& rep);
  Expression full_logits(const Expression& rep);
  /**
   * @brief Get log distribution over classes
   * 
   * @param rep Input vector
   * @return Vector of \f$\log(p(c\mid \texttt{rep}))\f$
   */
  Expression class_log_distribution(const Expression& rep);
  /**
   * @brief Get logits of classes
   * 
   * @param rep Input vector
   * @return Logits
   */
  Expression class_logits(const Expression& rep);
  /**
   * @brief Get log distribution over subclasses of class
   * 
   * @param rep Input vector
   * @param clusteridx Class index
   * 
   * @return Vector of \f$\log(p(i\mid c, \texttt{rep}))\f$
   */
  Expression subclass_log_distribution(const Expression& rep, unsigned clusteridx);
  /**
   * @brief Logits over subclasses of class
   * 
   * @param rep Input vector
   * @param clusteridx Class index
   * 
   * @return Logits
   */
  Expression subclass_logits(const Expression& rep, unsigned clusteridx);
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
    if (e.is_stale())
      e = update ? parameter(*pcg, p_rc2ws[cluster_idx]) : const_parameter(*pcg, p_rc2ws[cluster_idx]) ;
    return e;
  }
  inline Expression& get_rc2wbias(unsigned cluster_idx) {
    Expression& e = rc2biases[cluster_idx];
    if (e.is_stale())
      e =  update ? parameter(*pcg, p_rcwbiases[cluster_idx]) : const_parameter(*pcg, p_rcwbiases[cluster_idx]);
    return e;
  }
  ComputationGraph* pcg;
  Expression r2c;
  Expression cbias;
  std::vector<Expression> rc2ws;
  std::vector<Expression> rc2biases;
  bool bias;
  bool update;
};
}  // namespace dynet

#endif
