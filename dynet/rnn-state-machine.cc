#include "dynet/rnn-state-machine.h"

#include "dynet/except.h"

namespace dynet {

void RNNStateMachine::failure(RNNOp op) {
  DYNET_RUNTIME_ERR("State transition error: currently in state " << q_ << " but received operation " << op);
}

} // namespace dynet

