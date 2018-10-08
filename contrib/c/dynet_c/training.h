#ifndef DYNET_C_TRAINING_H_
#define DYNET_C_TRAINING_H_

#include <dynet_c/define.h>
#include <dynet_c/model.h>

/**
 * Opaque type of Trainer.
 */
typedef struct dynetTrainer dynetTrainer_t;

/**
 * Deletes the Trainer object.
 * @param trainer Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteTrainer(dynetTrainer_t *trainer);

/**
 * Updates parameters.
 * @param trainer Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetUpdateTrainer(dynetTrainer_t *trainer);

/**
 * Restarts the trainer.
 * @param trainer Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetRestartTrainer(dynetTrainer_t *trainer);

/**
 * Restarts the trainer with a new learning rate.
 * @param trainer Pointer of a handler.
 * @param lr New learning rate.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetRestartTrainerWithLearningRate(
    dynetTrainer_t *trainer, float lr);

/**
 * Prints information about the trainer.
 * @param trainer Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetPrintTrainerStatus(dynetTrainer_t *trainer);

/**
 * Gets global learning rate for all parameters.
 * @param trainer Pointer of a handler.
 * @param retval Pointer to receive the learning rate.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetTrainerLearningRate(
    const dynetTrainer_t *trainer, float *retval);

/**
 * Gets clipping threshold.
 * @param trainer Pointer of a handler.
 * @param retval Pointer to receive the clipping threshold.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetTrainerClipThreshold(
    const dynetTrainer_t *trainer, float *retval);

/**
 * Sets clipping threshold.
 * @param trainer Pointer of a handler.
 * @param threshold Clipping threshold.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetTrainerClipThreshold(
    dynetTrainer_t *trainer, float threshold);

/**
 * Creates a new SimpleSGDTrainer object.
 * @param m ParameterCollection to be trained.
 * @param learning_rate Initial learning rate.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateSimpleSGDTrainer(
    dynetParameterCollection_t *m, float learning_rate,
    dynetTrainer_t **newobj);

/**
 * Creates a new CyclicalSGDTrainer object.
 * @param m ParameterCollection to be trained.
 * @param learning_rate_min Lower learning rate.
 * @param learning_rate_max Upper learning rate.
 * @param step_size Period of the triangular function in number of iterations
 *                  (__not__ epochs).
 * @param gamma Learning rate upper bound decay parameter.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateCyclicalSGDTrainer(
    dynetParameterCollection_t *m, float learning_rate_min,
    float learning_rate_max, float step_size, float gamma,
    dynetTrainer_t **newobj);

/**
 * Creates a new MomentumSGDTrainer object.
 * @param m ParameterCollection to be trained.
 * @param learning_rate Initial learning rate.
 * @param mom Momentum.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateMomentumSGDTrainer(
    dynetParameterCollection_t *m, float learning_rate, float mom,
    dynetTrainer_t **newobj);

/**
 * Creates a new AdagradTrainer object.
 * @param m ParameterCollection to be trained.
 * @param learning_rate Initial learning rate.
 * @param eps Bias parameter \f$\epsilon\f$ in the adagrad formula.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateAdagradTrainer(
    dynetParameterCollection_t *m, float learning_rate, float eps,
    dynetTrainer_t **newobj);

/**
 * Creates a new AdadeltaTrainer object.
 * @param m ParameterCollection to be trained.
 * @param eps Bias parameter \f$\epsilon\f$ in the adagrad formula.
 * @param rho Update parameter for the moving average of updates in the
 *            numerator.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateAdadeltaTrainer(
    dynetParameterCollection_t *m, float eps, float rho,
    dynetTrainer_t **newobj);

/**
 * Creates a new RMSPropTrainer object.
 * @param m ParameterCollection to be trained.
 * @param learning_rate Initial learning rate.
 * @param eps Bias parameter \f$\epsilon\f$ in the adagrad formula.
 * @param rho Update parameter for the moving average (`rho = 0` is equivalent
 *            to using Adagrad).
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateRMSPropTrainer(
    dynetParameterCollection_t *m, float learning_rate, float eps, float rho,
    dynetTrainer_t **newobj);

/**
 * Creates a new AdamTrainer object.
 * @param m ParameterCollection to be trained.
 * @param learning_rate Initial learning rate.
 * @param beta_1 Moving average parameter for the mean.
 * @param beta_2 Moving average parameter for the variance.
 * @param eps Bias parameter \f$\epsilon\f$.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateAdamTrainer(
    dynetParameterCollection_t *m, float learning_rate, float beta_1,
    float beta_2, float eps, dynetTrainer_t **newobj);

/**
 * Creates a new AmsgradTrainer object.
 * @param m ParameterCollection to be trained.
 * @param learning_rate Initial learning rate.
 * @param beta_1 Moving average parameter for the mean.
 * @param beta_2 Moving average parameter for the variance.
 * @param eps Bias parameter \f$\epsilon\f$.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateAmsgradTrainer(
    dynetParameterCollection_t *m, float learning_rate, float beta_1,
    float beta_2, float eps, dynetTrainer_t **newobj);

#endif  // DYNET_C_TRAINING_H_
