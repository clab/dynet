#include "cnn/params.h"

namespace cnn {

ParametersBase::~ParametersBase() {}

size_t Parameters::size() const { return values.rows() * values.cols(); }

size_t LookupParameters::size() const {
  return values.size() * dim.rows * dim.cols;
}

} // namespace cnn

