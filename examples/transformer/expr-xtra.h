#pragma once

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"

#include <boost/range/irange.hpp>

using namespace dynet;

dynet::Expression arange(dynet::ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform, std::vector<float> *aux_mem);
dynet::Expression dither(dynet::ComputationGraph &cg, const dynet::Expression &expr, float pad_value, std::vector<float> *aux_mem);
dynet::Expression eq(const dynet::Expression &expr, float value, float epsilon=0.1);
dynet::Expression geq(const dynet::Expression &expr, float value, dynet::Expression &one, float epsilon=0.01); 
dynet::Expression leq(const dynet::Expression &expr, float value, dynet::Expression &one, float epsilon=0.01);
dynet::Expression softplus(const dynet::Expression &expr);

dynet::Expression layer_norm_colwise_2(const dynet::Expression& x, const dynet::Expression& g, const dynet::Expression& b, float epsilon=1e-8);
dynet::Expression layer_norm_colwise_3(const dynet::Expression& x, const dynet::Expression& g, const dynet::Expression& b);
dynet::Expression layer_norm_colwise_1(const dynet::Expression& x, const dynet::Expression& g, const dynet::Expression& b);

std::vector<dynet::Expression> split_rows(const dynet::Expression& x, unsigned h);
std::vector<dynet::Expression> split_batch(const dynet::Expression& x, unsigned h);

dynet::Expression make_time_distributed(const dynet::Expression& x);
dynet::Expression make_reverse_time_distributed(const dynet::Expression& x, unsigned int seq_len, unsigned int b_);

dynet::Expression arange(dynet::ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform, std::vector<float> *aux_mem) 
{
	aux_mem->clear();
	for (unsigned i = begin; i < end; ++i) 
		aux_mem->push_back((log_transform) ? log(1.0 + i) : i);
	return input(cg, dynet::Dim({end-begin}), aux_mem, dynet::default_device);
}

dynet::Expression repeat(dynet::ComputationGraph &cg, unsigned num, float value, std::vector<float> *aux_mem) 
{
	aux_mem->clear();
	aux_mem->resize(num, value);
	return input(cg, dynet::Dim({num}), aux_mem, dynet::default_device);
}

dynet::Expression dither(dynet::ComputationGraph &cg, const dynet::Expression &expr, float pad_value, std::vector<float> *aux_mem)
{
	const auto& shape = cg.nodes[expr.i]->dim;
	aux_mem->clear();
	aux_mem->resize(shape.cols(), pad_value);
	dynet::Expression padding = input(cg, dynet::Dim({shape.cols()}), aux_mem, dynet::default_device);
	dynet::Expression padded = dynet::concatenate(std::vector<dynet::Expression>({padding, expr, padding}));
	dynet::Expression left_shift = dynet::pickrange(padded, 2, shape.rows() + 2);
	dynet::Expression right_shift = dynet::pickrange(padded, 0, shape.rows());
	return dynet::concatenate_cols(std::vector<dynet::Expression>({left_shift, expr, right_shift}));
}

// binary boolean functions
dynet::Expression eq(const dynet::Expression &expr, float value, float epsilon) 
{
	return min(dynet::rectify(expr - (value - epsilon)), dynet::rectify(-expr + (value + epsilon))) / epsilon; 
}

dynet::Expression geq(const dynet::Expression &expr, float value, dynet::Expression &one, float epsilon) 
{
	return min(one, dynet::rectify(expr - (value - epsilon)) / epsilon);
}

dynet::Expression leq(const dynet::Expression &expr, float value, dynet::Expression &one, float epsilon) 
{
	return min(one, dynet::rectify((value + epsilon) - expr) / epsilon);
}

// @Vu -- this should be implemented in dynet!
dynet::Expression softplus(const dynet::Expression &expr) 
{
	return dynet::log(dynet::exp(expr) + 1);// https://www.tensorflow.org/api_docs/python/tf/nn/softplus
}

// @Vu: this layer_norm_colwise is an upgrade of dynet::layer_norm which only supports vector.
// Here, x can be either vector or matrix.
// refer to https://github.com/clab/dynet/issues/1066
dynet::Expression layer_norm_colwise_2(const dynet::Expression& x, const dynet::Expression& g, const dynet::Expression& b, float epsilon){
	dynet::Expression mu = dynet::transpose(dynet::mean_dim(x, {0}, true));
	mu = dynet::concatenate(std::vector<dynet::Expression>(x.dim()[0], mu));

	dynet::Expression sigma = dynet::transpose(dynet::std_dim(x, {0}, true));
	sigma = dynet::concatenate(std::vector<dynet::Expression>(x.dim()[0], sigma));
	
	dynet::Expression x_centered = x - mu;

	return dynet::cmult(g, dynet::cdiv(x_centered, sigma + epsilon)) + b;
}// version 2: a bit faster

dynet::Expression layer_norm_colwise_3(const dynet::Expression& x, const dynet::Expression& g, const dynet::Expression& b){
	Expression i_x_td = make_time_distributed(x);
	auto& d = x.dim();
	auto bs = d.batch_elems();
	return make_reverse_time_distributed(dynet::layer_norm(i_x_td, g, b), d[1], bs);
}// version 3: use time distributed trick?

dynet::Expression layer_norm_colwise_1(const dynet::Expression& x, const dynet::Expression& g, const dynet::Expression& b){
	std::vector<dynet::Expression> vCols(x.dim().d[1]);
	for (unsigned i = 0; i < x.dim().d[1]; i++){ 
		dynet::Expression c_x = dynet::select_cols(x, {i});
		vCols[i] = dynet::layer_norm(c_x, g, b);
	}
	return dynet::concatenate_cols(vCols);
}// version 1

std::vector<dynet::Expression> split_rows(const dynet::Expression& x, unsigned h){
	auto& d = x.dim();
	//auto b = d.batch_elems();
	
	unsigned steps = d[0] / h;
	
	std::vector<dynet::Expression> x_out;
	for (unsigned int i = 0; i < d[0]; i+=steps){
		x_out.push_back(dynet::pick_range(x, i, i + steps));
	}

	return x_out;
}

std::vector<dynet::Expression> split_batch(const dynet::Expression& x, unsigned h){
	auto& d = x.dim();
	unsigned b = d.batch_elems();

	std::vector<unsigned> l(b);
	std::iota(std::begin(l), std::end(l), 0);// fast way?

	unsigned steps = b / h;

	std::vector<dynet::Expression> output;
	for (unsigned i = 0; i < b; i+=steps){
		output.push_back(dynet::pick_batch_elems(x, std::vector<unsigned>(l.begin() + i, l.begin() + i + steps)));
	}

	return output;
}

dynet::Expression make_time_distributed(const dynet::Expression& x){
	auto& d = x.dim();
	auto b = d.batch_elems();		
	
	unsigned int total_words = d[1] * b;
	return dynet::reshape(x, dynet::Dim({d[0], 1}, total_words));
}

dynet::Expression make_reverse_time_distributed(const dynet::Expression& x, unsigned int seq_len, unsigned int b_){
	auto& d = x.dim();
	//auto b = d.batch_elems();
	//assert(seq_len * b_ == b);

	return dynet::reshape(x, dynet::Dim({d[0], seq_len}, b_));
}

