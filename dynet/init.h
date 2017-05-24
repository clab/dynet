#ifndef DYNET_EIGEN_INIT_H
#define DYNET_EIGEN_INIT_H

#include <string>
#include <vector>

namespace dynet {

extern float weight_decay_lambda;
extern int autobatch_flag;
extern int autobatch_debug_flag;

/**
 * \brief Represents general parameters for dynet
 *
 */
struct DynetParams {
  DynetParams();
  ~DynetParams();
  unsigned random_seed = 0; /**< The seed for random number generation */
  std::string mem_descriptor = "512"; /**< Total memory to be allocated for Dynet */
  float weight_decay = 0; /**< Weight decay rate for L2 regularization */
  int autobatch = 1; /**< Whether to autobatch or not */
  int autobatch_debug = 0; /**< Whether to show autobatch debug info or not */
  bool shared_parameters = false; /**< TO DOCUMENT */
  bool ngpus_requested = false; /**< GPUs requested by number */
  bool ids_requested = false; /**< GPUs requested by ids */
  int requested_gpus = -1; /**< Number of requested GPUs */
  std::vector<int> gpu_mask; /**< List of required GPUs by ids */


};

DynetParams extract_dynet_params(int& argc, char**& argv, bool shared_parameters = false);
void initialize(DynetParams& params);
void initialize(int& argc, char**& argv, bool shared_parameters = false);
void cleanup();

} // namespace dynet

#endif
