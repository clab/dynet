#ifndef DYNET_C_GRAPH_H_
#define DYNET_C_GRAPH_H_

#include <dynet_c/define.h>
#include <dynet_c/tensor.h>

typedef struct dynetExpression dynetExpression_t;

/**
 * Opaque type of ComputationGraph.
 */
typedef struct dynetComputationGraph dynetComputationGraph_t;

/**
 * Creates a new ComputationGraph object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateComputationGraph(
    dynetComputationGraph_t **newobj);

/**
 * Deletes the ComputationGraph object.
 * @param cg Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteComputationGraph(
    dynetComputationGraph_t *cg);

/**
 * Resets the ComputationGraph to a newly created state.
 * @param cg Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetClearComputationGraph(
    dynetComputationGraph_t *cg);

/**
 * Sets a checkpoint for the ComputationGraph.
 * @param cg Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetComputationGraphCheckpoint(
    dynetComputationGraph_t *cg);

/**
 * Reverts the ComputationGraph to the last checkpoint.
 * @param cg Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetRevertComputationGraph(
    dynetComputationGraph_t *cg);

/**
 * Runs complete forward pass from first node to given one, ignoring all
 * precomputed values.
 * @param cg Pointer of a handler.
 * @param last Expression up to which the forward pass must be computed.
 * @param retval Pointer to receive a calculated value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetForwardExprOnComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *last,
    const dynetTensor_t **retval);

/**
 * Runs forward pass from first node to given one.
 * @param cg Pointer of a handler.
 * @param last Expression up to which the forward pass must be computed.
 * @param retval Pointer to receive a calculated value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetForwardExprIncrementallyOnComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *last,
    const dynetTensor_t **retval);

/**
 * Get forward value for the given expression.
 * @param cg Pointer of a handler.
 * @param expr Expression to evaluate.
 * @param retval Pointer to receive a calculated value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetExprValueFromComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *expr,
    const dynetTensor_t **retval);

/**
 * Gets the gradient for the given expression.
 * @param cg Pointer of a handler.
 * @param expr Expression to evaluate.
 * @param retval Pointer to receive a calculated value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetExprGradientFromComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *expr,
    const dynetTensor_t **retval);

/**
 * Clears caches.
 * @param cg Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetInvalidateComputationGraph(
    dynetComputationGraph_t *cg);

/**
 * Computes backward gradients from the front-most evaluated node.
 * @param cg Pointer of a handler.
 * @param last Expression from which to compute the gradient.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetBackwardExprOnComputationGraph(
    dynetComputationGraph_t *cg, const dynetExpression_t *last);

/**
 * Visualizes the ComputationGraph.
 * @param cg Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetPrintComputationGraphViz(
    const dynetComputationGraph_t *cg);

/**
 * Dump the ComputationGraph to a file.
 * @param cg Pointer of a handler.
 * @param show_values Show internal values.
 * @param show_gradients Show gradient values.
 * @param nan_check_only Only check whether each gradient is nan.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDumpComputationGraph(
    dynetComputationGraph_t *cg, const char *filename,
    DYNET_C_BOOL show_values, DYNET_C_BOOL show_gradients,
    DYNET_C_BOOL nan_check_only);

#endif  // DYNET_C_GRAPH_H_
