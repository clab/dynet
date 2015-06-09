#include "cnn/expr.h"
#include "cnn/nodes.h"

namespace cnn { namespace expr {

Expression input(ComputationGraph& g, real s) { return Expression(&g, g.add_input(s)); }
Expression input(ComputationGraph& g, const real *ps) { return Expression(&g, g.add_input(ps)); }
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>* pdata) { return Expression(&g, g.add_input(d, pdata)); }
Expression parameter(ComputationGraph& g, Parameters* p) { return Expression(&g, g.add_parameters(p)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, unsigned index) { return Expression(&g, g.add_lookup(p, index)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex) { return Expression(&g, g.add_lookup(p, pindex)); }

Expression operator-(const Expression& x) { return Expression(x.pg, x.pg->add_function<Negate>({x.i})); }
Expression operator+(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Sum>({x.i, y.i})); }
Expression operator-(const Expression& x, const Expression& y) { return x+(-y); }
Expression operator*(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<MatrixMultiply>({x.i, y.i})); }
    
Expression tanh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Tanh>({x.i})); }
Expression logistic(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogisticSigmoid>({x.i})); }
Expression rectify(const Expression& x) { return Expression(x.pg, x.pg->add_function<Rectify>({x.i})); }
Expression log_softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogSoftmax>({x.i})); }

Expression cwise_multiply(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<CwiseMultiply>({x.i, y.i}));}

Expression concatenate(const std::vector<Expression>& x) {
	std::vector<VariableIndex> tmp(x.size());
	for(unsigned j = 0; j < x.size(); ++j){
		tmp[j] = x[j].i;
	}
	//ComputationGraph *tpg = x[0].pg;
	return Expression(x[0].pg, x[0].pg->add_function<Concatenate>(tmp)); 
}
Expression sum(const std::vector<Expression>& x) {
	std::vector<VariableIndex> tmp(x.size());
	for(unsigned j = 0; j < x.size(); ++j){
		tmp[j] = x[j].i;
	}
	//tpg = x[0].pg;
	return Expression(x[0].pg, x[0].pg->add_function<Sum>(tmp)); 
}

Expression squaredDistance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<SquaredEuclideanDistance>({x.i, y.i})); }
Expression pick(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, v)); }
Expression pickrange(const Expression& x, unsigned v, unsigned u) { return Expression(x.pg, x.pg->add_function<PickRange>({x.i}, v, u)); }

} }
