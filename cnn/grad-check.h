#ifndef CNN_GRAD_CHECK_H
#define CNN_GRAD_CHECK_H

#include "cnn/expr.h"

namespace cnn {

class Model;
struct ComputationGraph;

// verbosity is zero for silence, one for only printing errors, two for everything
bool check_grad(Model& m, expr::Expression& expr, int verbosity = 1);

} // namespace cnn

#endif
