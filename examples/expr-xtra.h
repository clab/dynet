#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

using namespace cnn;

// Chris -- this should be a library function
Expression arange(ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform) 
{
    auto * as = new std::vector<float>(end-begin); // FIXME: memory leak
    for (unsigned i = begin; i < end; ++i) 
        as->push_back((log_transform) ? log(1.0 + i) : i);
    long dist = end - begin; 
    return Expression(&cg, cg.add_input(Dim({dist}), as));
}

// Chris -- this should be a library function
Expression repeat(ComputationGraph &cg, unsigned num, float value) 
{
    auto* rep = new std::vector<float>(num, value); // FIXME: memory leak
    return Expression(&cg, cg.add_input(Dim({long(num)}), rep));
}

// Chris -- this should be a library function
Expression dither(ComputationGraph &cg, const Expression &expr, float pad_value=0.0)
{
    const auto& shape = cg.nodes[expr.i]->dim;
    auto* pad = new std::vector<float>(shape.cols(), pad_value); // FIXME: memory leak
    Expression padding(&cg, cg.add_input(Dim({shape.cols()}), pad));
    Expression padded = concatenate(std::vector<Expression>({padding, expr, padding}));
    Expression left_shift = pickrange(padded, 2, shape.rows()+2);
    Expression right_shift = pickrange(padded, 0, shape.rows());
    return concatenate_cols(std::vector<Expression>({left_shift, expr, right_shift}));
}

// these expressions can surely be implemented much more efficiently than this
Expression abs(const Expression &expr) 
{
    return rectify(expr) + rectify(-expr); 
}

// binary boolean functions, is it better to use a sigmoid?
Expression eq(const Expression &expr, float value, float epsilon=0.1) 
{
    return min(rectify(expr - (value - epsilon)), rectify(-expr + (value + epsilon))) / epsilon; 
}

Expression geq(const Expression &expr, float value, Expression &one, float epsilon=0.01) 
{
    return min(one, rectify(expr - (value - epsilon)) / epsilon);
        //rectify(1 - rectify(expr - (value - epsilon)));
}

Expression leq(const Expression &expr, float value, Expression &one, float epsilon=0.01) 
{
    return min(one, rectify((value + epsilon) - expr) / epsilon);
    //return rectify(1 - rectify((value + epsilon) - expr));
}
