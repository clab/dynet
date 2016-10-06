#include "cnn/weight-decay.h"
#include "cnn/io-macros.h"

namespace cnn {

template<class Archive>
void L2WeightDecay::serialize(Archive& ar, const unsigned int) {
  ar & weight_decay;
  ar & lambda;
}
CNN_SERIALIZE_IMPL(L2WeightDecay)

}
