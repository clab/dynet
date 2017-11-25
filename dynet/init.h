#ifndef DYNET_EIGEN_INIT_H
#define DYNET_EIGEN_INIT_H

#include <string>
#include <vector>

namespace dynet {

extern float weight_decay_lambda;
extern int autobatch_flag;
extern int profiling_flag;

/**
 * \brief Represents general parameters for dynet
 *
 */
struct DynetParams {
  DynetParams();
  ~DynetParams();
  unsigned random_seed; /**< The seed for random number generation */
  std::string mem_descriptor; /**< Total memory to be allocated for Dynet */
  float weight_decay; /**< Weight decay rate for L2 regularization */
  int autobatch; /**< Whether to autobatch or not */
  int profiling; /**< Whether to show autobatch debug info or not */
  bool shared_parameters; /**< TO DOCUMENT */
  bool ngpus_requested; /**< GPUs requested by number */
  bool ids_requested; /**< GPUs requested by ids */
  bool cpu_requested; /**< CPU requested in multi-device case */
  int requested_gpus; /**< Number of requested GPUs */
  std::vector<int> gpu_mask; /**< List of required GPUs by ids */
};

DynetParams extract_dynet_params(int& argc, char**& argv, bool shared_parameters = false);
void initialize(DynetParams& params);
void initialize(int& argc, char**& argv, bool shared_parameters = false);
void cleanup();

} // namespace dynet

#endif
