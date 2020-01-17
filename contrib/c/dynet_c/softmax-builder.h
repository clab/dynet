#ifndef DYNET_C_SOFTMAX_BUILDER_H_
#define DYNET_C_SOFTMAX_BUILDER_H_

#include <dynet_c/define.h>
#include <dynet_c/expr.h>
#include <dynet_c/graph.h>
#include <dynet_c/model.h>

/**
 * Opaque type of SoftmaxBuilder.
 */
typedef struct dynetSoftmaxBuilder dynetSoftmaxBuilder_t;

/**
 * Deletes the SoftmaxBuilder object.
 * @param builder Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteSoftmaxBuilder(
    dynetSoftmaxBuilder_t *builder);

/**
 * Initializes the parameters in the computation graph.
 * @param builder Pointer of a handler.
 * @param h_0 Vector to initialize hidden layers at timestep 0.
 * @param n The number of `h_0` elements.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetResetSoftmaxBuilderGraph(
    dynetSoftmaxBuilder_t *builder, dynetComputationGraph_t *cg,
    DYNET_C_BOOL update);

/**
 * Computes negative log probability of a class.
 * @param builder Pointer of a handler.
 * @param rep Vector expression.
 * @param classidx Class.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySoftmaxBuilderNegLogSoftmaxOne(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    uint32_t classidx, dynetExpression_t **newobj);

/**
 * Computes batched negative log probability of a class.
 * @param builder Pointer of a handler.
 * @param rep Vector expression (batched).
 * @param classidxs List of classes, one per batch element.
 * @param n Number of indices.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySoftmaxBuilderNegLogSoftmax(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    const uint32_t *classidxs, size_t n, dynetExpression_t **newobj);

/**
 * Samples from the softmax distribution.
 * @param builder Pointer of a handler.
 * @param rep Vector expression parametrizing the distribution.
 * @param retval Pointer to receive a sampled class.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSampleFromSoftmaxBuilder(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    uint32_t *retval);

/**
 * Returns an Expression representing a vector the size of the number of
 * classes.
 * @param rep Vector expression parametrizing the distribution.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetSoftmaxBuilderFullLogDistribution(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    dynetExpression_t **newobj);

/**
 * Returns the logits (before application of the softmax).
 * @param rep Vector expression parametrizing the distribution.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetSoftmaxBuilderFullLogits(
    dynetSoftmaxBuilder_t *builder, const dynetExpression_t *rep,
    dynetExpression_t **newobj);

/**
 * Gets the ParameterCollection containing the softmax parameters.
 * @param builder Pointer of a handler.
 * @param newobj Pointer to receive the parameter collection.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetSoftmaxBuilderParameterCollection(
    dynetSoftmaxBuilder_t *builder, dynetParameterCollection_t **newobj);

/**
 * Creates a new StandardSoftmaxBuilder object.
 * @param rep_dim Dimension of the input vectors.
 * @param num_classes Number of classes.
 * @param pc Parameter collection.
 * @param bias Whether to use a bias vector or not.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateStandardSoftmaxBuilder(
    uint32_t rep_dim, uint32_t num_classes, dynetParameterCollection_t *pc,
    DYNET_C_BOOL bias, dynetSoftmaxBuilder_t **newobj);

/**
 * Creates a new StandardSoftmaxBuilder object with pre-existing parameters.
 * @param p_w Weight matrix.
 * @param p_b Bias vector (no bias is used if `p_b` is nullptr).
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateStandardSoftmaxBuilderFromParameters(
    dynetParameter_t *p_w, dynetParameter_t *p_b,
    dynetSoftmaxBuilder_t **newobj);

#endif  // DYNET_C_SOFTMAX_BUILDER_H_
