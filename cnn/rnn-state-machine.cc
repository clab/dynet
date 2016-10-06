#include "cnn/rnn-state-machine.h"

#include <iostream>

#include "cnn/cnn.h"
#include "cnn/io-macros.h"

using namespace std;

namespace cnn {

void RNNStateMachine::failure(RNNOp op) {
  cerr << "State transition error: currently in state " << q_ << " but received operation " << op << endl;
  abort();
}

template <class Archive>
void RNNStateMachine::serialize(Archive& ar, const unsigned int) {
  ar & q_;
}
CNN_SERIALIZE_IMPL(RNNStateMachine)

} // namespace cnn

