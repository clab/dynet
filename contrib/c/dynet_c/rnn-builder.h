#ifndef DYNET_C_RNN_BUILDER_H_
#define DYNET_C_RNN_BUILDER_H_

#include <dynet_c/define.h>
#include <dynet_c/graph.h>
#include <dynet_c/model.h>

/**
 * Opaque type of RNNBuilder.
 */
typedef struct dynetRNNBuilder dynetRNNBuilder_t;

/**
 * Deletes the RNNBuilder object.
 * @param builder Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteRNNBuilder(dynetRNNBuilder_t *builder);

/**
 * Gets pointer to the current state.
 * @param builder Pointer of a handler.
 * @param retval Pointer to receive the pointer.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetRNNBuilderStatePointer(
    const dynetRNNBuilder_t *builder, int32_t *retval);

/**
 * Resets the internally used computation graph with a new one.
 * @param builder Pointer of a handler.
 * @param cg Computation graph.
 * @param update Update internal parameters while training.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetResetRNNBuilderGraph(
    dynetRNNBuilder_t *builder, dynetComputationGraph_t *cg,
    DYNET_C_BOOL update);

/**
 * Resets the builder for a new sequence.
 * @param builder Pointer of a handler.
 * @param h_0 Vector to initialize hidden layers at timestep 0.
 * @param n The number of `h_0` elements.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetStartRNNBuilderNewSequence(
    dynetRNNBuilder_t *builder, const dynetExpression_t *const *h_0, size_t n);

/**
 * Sets the output state of a node.
 * @param builder Pointer of a handler.
 * @param prev Pointer to the previous state.
 * @param h_new The new hidden state.
 * @param n The number of `h_new` elements.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetRNNBuilderHiddenState(
    dynetRNNBuilder_t *builder, int32_t prev,
    const dynetExpression_t *const *h_new, size_t n);

/**
 * Sets the internal state of a node.
 * @param builder Pointer of a handler.
 * @param prev Pointer to the previous state.
 * @param c_new The new cell state.
 * @param n The number of `c_new` elements.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetRNNBuilderCellState(
    dynetRNNBuilder_t *builder, int32_t prev,
    const dynetExpression_t *const *c_new, size_t n);

/**
 * Adds another timestep by reading in the variable x.
 * @param builder Pointer of a handler.
 * @param x Input variable.
 * @param newobj Pointer to receive the hidden representation.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetAddRNNBuilderInput(
    dynetRNNBuilder_t *builder, const dynetExpression_t *x,
    dynetExpression_t **newobj);

/**
 * Adds another timestep, with arbitrary recurrent connection.
 * @param builder Pointer of a handler.
 * @param prev Pointer to the previous state.
 * @param x Input variable.
 * @param newobj Pointer to receive the hidden representation.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetAddRNNBuilderInputToState(
    dynetRNNBuilder_t *builder, int32_t prev, const dynetExpression_t *x,
    dynetExpression_t **newobj);

/**
 * Rewinds the last timestep.
 * @param builder Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetRewindRNNBuilderOneStep(
    dynetRNNBuilder_t *builder);

/**
 * Returns the RNN state that is the parent of the given state.
 * @param builder Pointer of a handler.
 * @param p Pointer to a state.
 * @param retval Pointer to receive the pointer.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetRNNBuilderParentStatePointer(
    const dynetRNNBuilder_t *builder, int32_t p, int32_t *retval);

/**
 * Sets dropout.
 * @param builder Pointer of a handler.
 * @param d Dropout rate.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetRNNBuilderDropout(
    dynetRNNBuilder_t *builder, float d);

/**
 * Disables dropout.
 * @param builder Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDisableRNNBuilderDropout(
    dynetRNNBuilder_t *builder);

/**
 * Returns node (index) of most recent output.
 * @param builder Pointer of a handler.
 * @param newobj Pointer to receive the output.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetRNNBuilderLastOutput(
    const dynetRNNBuilder_t *builder, dynetExpression_t **newobj);

/**
 * Returns the final output of each hidden layer.
 * @param builder Pointer of a handler.
 * @param newobj Pointer of an array to receive the output.
 * @param size Pointer to receive the number of the output.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetRNNBuilderFinalHiddenState(
    const dynetRNNBuilder_t *builder, dynetExpression_t **newobj,
    size_t *size);

/**
 * Returns the output of any hidden layer.
 * @param builder Pointer of a handler.
 * @param i Pointer to the step which output you want to access.
 * @param newobj Pointer of an array to receive the output.
 * @param size Pointer to receive the number of the output.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetRNNBuilderHiddenState(
    const dynetRNNBuilder_t *builder, int32_t i, dynetExpression_t **newobj,
    size_t *size);

/**
 * Returns the final state of each hidden layer.
 * @param builder Pointer of a handler.
 * @param newobj Pointer of an array to receive the state.
 * @param size Pointer to receive the number of the state.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetRNNBuilderFinalCellState(
    const dynetRNNBuilder_t *builder, dynetExpression_t **newobj,
    size_t *size);

/**
 * Returns the state of any hidden layer.
 * @param builder Pointer of a handler.
 * @param i Pointer to the step which state you want to access.
 * @param newobj Pointer of an array to receive the state.
 * @param size Pointer to receive the number of the state.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetRNNBuilderCellState(
    const dynetRNNBuilder_t *builder, int32_t i, dynetExpression_t **newobj,
    size_t *size);

/**
 * Returns the number of components in `h_0`.
 * @param builder Pointer of a handler.
 * @param retval Pointer to receive the number.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetRNNBuilderNumH0Components(
    const dynetRNNBuilder_t *builder, int32_t *retval);

/**
 * Copies the parameters of another builder.
 * @param builder Pointer of a handler.
 * @param params Source RNNBuilder to copy parameters from.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCopyRNNBuilderParameters(
    dynetRNNBuilder_t *builder, const dynetRNNBuilder_t *params);

/**
 * Gets parameters in the RNNBuilder.
 * @param builder Pointer of a handler.
 * @param newobj Pointer to receive the parameter collection.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetRNNBuilderParameterCollection(
    dynetRNNBuilder_t *builder, dynetParameterCollection_t **newobj);

/**
 * Creates a new SimpleRNNBuilder object.
 * @param layers Number of layers.
 * @param input_dim Dimension of the input.
 * @param hidden_dim Hidden layer (and output) size.
 * @param model ParameterCollection holding the parameters.
 * @param support_lags Allow auxiliary output or not.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateSimpleRNNBuilder(
    uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, DYNET_C_BOOL support_lags,
    dynetRNNBuilder_t **newobj);

/**
 * Adds auxiliary output.
 * @param builder Pointer of a handler.
 * @param x Input expression.
 * @param aux Auxiliary output expression.
 * @param newobj Pointer to receive the hidden representation.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetAddSimpleRNNBuilderAuxiliaryInput(
    dynetRNNBuilder_t *builder, const dynetExpression_t *x,
    const dynetExpression_t *aux, dynetExpression_t **newobj);

/**
 * Sets dropout.
 * @param builder Pointer of a handler.
 * @param d Dropout rate.
 * @param d_h Another dropout rate.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetSimpleRNNBuilderDropout(
    dynetRNNBuilder_t *builder, float d, float d_h);

/**
 * Set dropout masks at the beginning of a sequence for a specific batch size.
 * @param builder Pointer of a handler.
 * @param batch_size Batch size.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetSimpleRNNBuilderDropoutMasks(
    dynetRNNBuilder_t *builder, uint32_t batch_size);

#endif  // DYNET_C_RNN_BUILDER_H_
