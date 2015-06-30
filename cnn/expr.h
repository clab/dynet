#ifndef EXPR_H
#define EXPR_H

#include "cnn/cnn.h"
#include "cnn/nodes.h"

namespace cnn { namespace expr {

struct Expression {
  ComputationGraph *pg;
  VariableIndex i;

  Expression() : pg(nullptr) { }
  Expression(ComputationGraph *pg, VariableIndex i) : pg(pg), i(i) { }
};

Expression input(ComputationGraph& g, real s);
Expression input(ComputationGraph& g, const real *ps);
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>* pdata);
Expression parameter(ComputationGraph& g, Parameters* p);
Expression lookup(ComputationGraph& g, LookupParameters* p, unsigned index);
Expression lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex);
Expression const_lookup(ComputationGraph& g, LookupParameters* p, unsigned index);
Expression const_lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex);

Expression operator-(const Expression& x);
Expression operator+(const Expression& x, const Expression& y);
//Expression operator+(const Expression& x, real y);
Expression operator-(const Expression& x, const Expression& y);
Expression operator-(real x, const Expression& y);
Expression operator*(const Expression& x, const Expression& y);
Expression operator*(const Expression& x, float y);
inline Expression operator*(float y, const Expression& x) { return x * y; }
inline Expression operator/(const Expression& x, float y) { return x * (1.f / y); }
// componentwise division
Expression cdiv(const Expression& x, const Expression& y);
Expression colwise_add(const Expression& x, const Expression& bias);

Expression tanh(const Expression& x);
Expression exp(const Expression& x);
Expression log(const Expression& x);
Expression logistic(const Expression& x);
Expression rectify(const Expression& x);
Expression hinge(const Expression& x, unsigned index, float m = 1.0);
Expression hinge(const Expression& x, const unsigned* pindex, float m = 1.0);
Expression log_softmax(const Expression& x);
Expression log_softmax(const Expression& x, const std::vector<unsigned>& restriction);
Expression softmax(const Expression& x);
Expression softsign(const Expression& x);
Expression noise(const Expression& x, real stddev);
Expression dropout(const Expression& x, real p);

Expression reshape(const Expression& x, const Dim& d);
Expression transpose(const Expression& x);

Expression affine_transform(const std::initializer_list<Expression>& xs);
Expression affine_transform(ComputationGraph& g, const std::vector<VariableIndex>& xs);
Expression cwise_multiply(const Expression& x, const Expression& y);

Expression dot_product(const Expression& x, const Expression& y);
Expression squared_distance(const Expression& x, const Expression& y);
Expression huber_distance(const Expression& x, const Expression& y, float c = 1.345f);
Expression l1_distance(const Expression& x, const Expression& y);
Expression binary_log_loss(const Expression& x, const Expression& y);
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m=1.0);

Expression conv1d_narrow(const Expression& x, const Expression& f);
Expression conv1d_wide(const Expression& x, const Expression& f);
Expression kmax_pooling(const Expression& x, unsigned k);
Expression fold_rows(const Expression& x, unsigned nrows=2);

Expression pick(const Expression& x, unsigned v);
Expression pick(const Expression& x, unsigned* pv);
Expression pickrange(const Expression& x, unsigned v, unsigned u);

Expression pickneglogsoftmax(const Expression& x, unsigned v);

template <typename T>
Expression sum(const T& xs) {
  ComputationGraph *pg = xs.begin()->pg;
  std::vector<VariableIndex> xis(xs.size());
  for (int i=0; i<xs.size(); ++i) xis[i] = xs[i].i;
  return Expression(pg, pg->add_function<Sum>(xis));
}

template <typename T>
Expression average(const T& xs) {
  ComputationGraph *pg = xs.begin()->pg;
  std::vector<VariableIndex> xis(xs.size());
  for (int i=0; i<xs.size(); ++i) xis[i] = xs[i].i;
  return Expression(pg, pg->add_function<Average>(xis));
}

template <typename T>
Expression concatenate_cols(const T& xs) {
  ComputationGraph *pg = xs.begin()->pg;
  std::vector<VariableIndex> xis(xs.size());
  for (int i=0; i<xs.size(); ++i) xis[i] = xs[i].i;
  return Expression(pg, pg->add_function<ConcatenateColumns>(xis));
}

template <typename T>
Expression concatenate(const T& xs) {
  ComputationGraph *pg = xs.begin()->pg;
  std::vector<VariableIndex> xis(xs.size());
  for (int i=0; i<xs.size(); ++i) xis[i] = xs[i].i;
  return Expression(pg, pg->add_function<Concatenate>(xis));
}

Expression sum_cols(const Expression& x);

Expression kmh_ngram(const Expression& x, unsigned n);

} }

#endif
