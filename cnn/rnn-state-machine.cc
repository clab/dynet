#include "cnn/rnn-state-machine.h"

#include <iostream>

using namespace std;

namespace cnn {

void RNNStateMachine::failure(RNNOp op) {
  cerr << "State transition error: currently in state " << q_ << " but received operation " << op << endl;
  abort();
}

} // namespace cnn

