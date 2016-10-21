#ifndef DYNET_EIGEN_INIT_H
#define DYNET_EIGEN_INIT_H

#include <string>
#if HAVE_CUDA
#include "dynet/cuda.h"
#endif

namespace dynet {

extern float weight_decay_lambda;

/**
 * \brief Represents general parameters for dynet
 *
 */
struct DynetParams {
  unsigned random_seed = 0; /**< The seed for random number generation */
  std::string mem_descriptor = "512"; /**< Total memory to be allocated for Dynet */
  float weight_decay = 0; /**< Weight decay rate for L2 regularization */
  bool shared_parameters = false; /**< TO DOCUMENT */

#if HAVE_CUDA
  bool ngpus_requested = false; /**< GPUs requested by number */
  bool ids_requested = false; /**< GPUs requested by ids */
  int requested_gpus = -1; /**< Number of requested GPUs */
  vector<int> gpu_mask(MAX_GPUS, 0); /**< List of required GPUs by ids */
#endif

};

DynetParams extract_dynet_params(int& argc, char**& argv, bool shared_parameters = false);
void initialize(DynetParams params);
void initialize(int& argc, char**& argv, bool shared_parameters = false);
void cleanup();

} // namespace dynet

#endif
