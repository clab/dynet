#pragma once

// DyNet
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"

// STL
#include <vector>
#include <limits>

using namespace std;
using namespace dynet;

#define PSEUDO_MIN_VALUE -999999.f

dynet::Expression create_triangle_mask(dynet::ComputationGraph &cg, unsigned length, bool upper=true/*false for lower*/);

dynet::Expression create_triangle_mask(dynet::ComputationGraph &cg, unsigned length, bool upper) {
	// fill triangle mask
	std::vector<float> vMask(length * length, 0.f);
	for(unsigned i = 0; i < length; ++i){
		for(unsigned j = 0; j <= i; ++j){
			if (!upper)// lower
				vMask[i * length + j] = 1.f;
			else// upper
				vMask[j * length + i] = 1.f;
		}
	}
	
	dynet::Expression i_mask = dynet::input(cg, {length, length}, vMask);

	i_mask = (1.f - i_mask) * PSEUDO_MIN_VALUE;// convert 0/1 mask to transformer style -inf mask

	return i_mask;
}

