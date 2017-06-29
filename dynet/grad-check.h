#ifndef DYNET_GRAD_CHECK_H
#define DYNET_GRAD_CHECK_H

#include "dynet/expr.h"

namespace dynet {

class ParameterCollection;
struct ComputationGraph;

// verbosity is zero for silence, one for only printing errors, two for everything
bool check_grad(ParameterCollection& m, Expression& expr, int verbosity = 1);

} // namespace dynet

#endif
