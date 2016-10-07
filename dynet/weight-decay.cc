#include "dynet/weight-decay.h"
#include "dynet/io-macros.h"

namespace dynet {

template<class Archive>
void L2WeightDecay::serialize(Archive& ar, const unsigned int) {
  ar & weight_decay;
  ar & lambda;
}
DYNET_SERIALIZE_IMPL(L2WeightDecay)

}
