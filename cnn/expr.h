#ifndef EXPR_H
#define EXPR_H

#include "cnn/cnn.h"

namespace cnn { namespace expr {

struct Expression {
  ComputationGraph *pg;
  VariableIndex i;

  Expression(ComputationGraph *pg, VariableIndex i) : pg(pg), i(i) { }
};

Expression input(ComputationGraph& g, real s);
Expression input(ComputationGraph& g, const real *ps);
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>* pdata);
Expression parameter(ComputationGraph& g, Parameters* p);
Expression lookup(ComputationGraph& g, LookupParameters* p, unsigned index);
Expression lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex);

Expression operator-(const Expression& x);
Expression operator+(const Expression& x, const Expression& y);
Expression operator-(const Expression& x, const Expression& y);
Expression operator*(const Expression& x, const Expression& y);
//Expression cwiseMultiply(const Expression& x, const Expression& y);

Expression tanh(const Expression& x);
Expression logistic(const Expression& x);
Expression rectify(const Expression& x);
Expression log_softmax(const Expression& x);

Expression squaredDistance(const Expression& x, const Expression& y);
Expression pick(const Expression& x, unsigned v);

} }

#endif
