#ifndef _NCE_H_
#define _NCE_H_

#include <algorithm>
#include <vector>
#include <numeric>

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/param-init.h"

using namespace dynet;
using namespace std;

/**
 * Interface for a builder that handles both the output layer and the
 * the softmax (or approximation thereof)
 */
class LossBuilder {
public:
  virtual Expression fsm_loss(ComputationGraph& cg, Expression h, const vector<unsigned>& ytrue) = 0;
  virtual Expression nce_loss(ComputationGraph& cg, Expression h,
                              const vector<unsigned>& ytrue_indx, const vector<float>& ytrue_prob,
                              const vector<unsigned>& sample_indx, const vector<float>& sample_cnt,
                              const vector<float>& sample_prob) = 0;
};

/**
 * Encapsulates the output layer of a multiclass classifer with a fixed
 * vocabulary size, and provides an efficient NCE loss that approximates
 * the full softmax loss.
 * 
 * Also provides a slighly inefficient exact fullsoftmax loss, useful
 * for evaluating your NCE models.
 */
class NCELossBuilder : public LossBuilder {
private:
  LookupParameter p_oX, p_ob;

public:
  NCELossBuilder(unsigned vocabSize, unsigned hDim, ParameterCollection& model, float initWidth=0.1) {
    // Note that the output layer is represented with lookup parameters
    // for sparse efficiency
    p_oX = model.add_lookup_parameters(vocabSize, {hDim}, ParameterInitUniform(initWidth));
    // Need to initialize output b's to be self-normalized
    p_ob = model.add_lookup_parameters(vocabSize, {1}, ParameterInitConst(-log(vocabSize)));
  }

  Expression fsm_loss(ComputationGraph& cg, Expression h, const vector<unsigned>& ytrue) {
    Expression oX = transpose(parameter(cg, p_oX)); // {O,H}
    Expression ob = transpose(parameter(cg, p_ob)); // {O}
    Expression o = ob + oX * h;                     // {OxB}
    Expression loss = pickneglogsoftmax(o, ytrue);  // {1xB}
    return loss;
  }

  Expression nce_loss(ComputationGraph& cg, Expression h,
                      const vector<unsigned>& ytrue_indx, const vector<float>& ytrue_prob,
                      const vector<unsigned>& sample_indx, const vector<float>& sample_cnt,
                      const vector<float>& sample_prob) {
    // Calculate k from sample_cnt
    //
    const float k = accumulate(sample_cnt.begin(), sample_cnt.end(), 0);
    
    // Build up a sample matrix from the lookup table
    //
    vector<Expression> sampled_oX_rows;
    vector<Expression> sampled_ob;
    sampled_oX_rows.reserve(sample_indx.size());
    sampled_ob.reserve(sample_indx.size());
    for(unsigned indx : sample_indx) {
      sampled_oX_rows.push_back(transpose(lookup(cg, p_oX, indx)));
      sampled_ob.push_back(lookup(cg, p_ob, indx));
    }
    Expression oX_sample = concatenate(sampled_oX_rows); // {S,H}
    Expression ob_sample = concatenate(sampled_ob);      // {S}

    // Calculate the noise half of the NCE equation
    //
    Expression o_sample = ob_sample + oX_sample * h;     // {SxB}
    Expression probs_sample = input(cg, {unsigned(sample_prob.size())}, sample_prob); // {S}
    Expression log_p_neg_noise = log(logistic(-delta_s(o_sample,probs_sample,k)));    // {SxB}
    Expression cnts_sample = input(cg, {unsigned(sample_cnt.size())}, sample_cnt);    // {S}
    Expression sum_log_p_neg_noise = dot_product(cnts_sample, log_p_neg_noise);       // {1xB}

    // Calculate the data half of the NCE equation
    //
    Expression oX_ytrue = lookup(cg, p_oX, ytrue_indx); // {HxB}
    Expression ob_ytrue = lookup(cg, p_ob, ytrue_indx); // {1xB}
    Expression o_ytrue = ob_ytrue + dot_product(oX_ytrue,  h); // Sparse dot product of vectors: {1xB}
    Expression probs_ytrue = input(cg, Dim({1},ytrue_indx.size()), ytrue_prob); // {1xB}
    Expression log_p_pos_data = log(logistic(delta_s(o_ytrue,probs_ytrue,k)));  // {1xB}

    // Final loss combines the two
    //
    Expression loss = -log_p_pos_data - sum_log_p_neg_noise;
    return loss;
  }
  
private:
  
  // Pre-activation score for probability of data given word and context
  //
  Expression delta_s(Expression scores, Expression probs, float k) {
    return scores - log(k * probs);
  }
  
};

/**
 * Builds an efficient traditional full softmax using the interface to our NCE
 * output builder
 */
class FSMLossBuilder : public LossBuilder {
private:
  Parameter W_sm;
  Parameter b_sm;

public:
  FSMLossBuilder(unsigned vocabSize, unsigned hDim, ParameterCollection& model, float initWidth=0.5) {
    W_sm = model.add_parameters({vocabSize, hDim}, ParameterInitUniform(initWidth));
    b_sm = model.add_parameters({vocabSize}, ParameterInitUniform(initWidth));
  }

  Expression fsm_loss(ComputationGraph& cg, Expression h, const vector<unsigned>& ytrue) {
    Expression W_exp = parameter(cg, W_sm);
    Expression b_exp = parameter(cg, b_sm);
    Expression score = affine_transform({b_exp, W_exp, h});
    Expression loss = pickneglogsoftmax(score, ytrue);
    return loss;
  }

  Expression nce_loss(ComputationGraph& cg, Expression h,
                      const vector<unsigned>& ytrue_indx, const vector<float>& ytrue_prob,
                      const vector<unsigned>& sample_indx, const vector<float>& sample_cnt,
                      const vector<float>& sample_prob) {
    return fsm_loss(cg, h, ytrue_indx);
  } 
};

#endif
