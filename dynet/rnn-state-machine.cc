#include "dynet/rnn-state-machine.h"

#include <iostream>

#include "dynet/dynet.h"
#include "dynet/io-macros.h"

using namespace std;

namespace dynet {

void RNNStateMachine::failure(RNNOp op) {
  cerr << "State transition error: currently in state " << q_ << " but received operation " << op << endl;
  abort();
}

template <class Archive>
void RNNStateMachine::serialize(Archive& ar, const unsigned int) {
  ar & q_;
}
DYNET_SERIALIZE_IMPL(RNNStateMachine)

} // namespace dynet

