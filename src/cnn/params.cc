#include "cnn/params.h"

namespace cnn {

ParametersBase::~ParametersBase() {}

size_t Parameters::size() const { return values.rows() * values.cols(); }

// since these aren't optimized, don't count them
size_t ConstParameters::size() const { return 0; }

size_t LookupParameters::size() const {
  return values.size() * dim.rows * dim.cols;
}

} // namespace cnn

