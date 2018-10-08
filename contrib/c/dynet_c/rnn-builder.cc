#include <dynet_c/config.h>

#include <dynet/fast-lstm.h>
#include <dynet/gru.h>
#include <dynet/lstm.h>
#include <dynet/rnn.h>
#include <dynet/treelstm.h>
#include <dynet_c/internal.h>
#include <dynet_c/rnn-builder.h>

#include <vector>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;
using dynet_c::internal::to_c_ptr_from_value;

DYNET_C_STATUS dynetDeleteRNNBuilder(dynetRNNBuilder_t *builder) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  delete to_cpp_ptr(builder);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetRNNBuilderStatePointer(
    const dynetRNNBuilder_t *builder, int32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(builder)->state();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetResetRNNBuilderGraph(
    dynetRNNBuilder_t *builder, dynetComputationGraph_t *cg,
    DYNET_C_BOOL update) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(cg);
  to_cpp_ptr(builder)->new_graph(*to_cpp_ptr(cg), update);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetStartRNNBuilderNewSequence(
    dynetRNNBuilder_t *builder, const dynetExpression_t *const *h_0,
    size_t n) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(h_0);
  std::vector<dynet::Expression> _h_0;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(h_0, n, &_h_0);
  to_cpp_ptr(builder)->start_new_sequence(_h_0);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetRNNBuilderHiddenState(
    dynetRNNBuilder_t *builder, int32_t prev,
    const dynetExpression_t *const *h_new, size_t n) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(h_new);
  std::vector<dynet::Expression> _h_new;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(h_new, n, &_h_new);
  to_cpp_ptr(builder)->set_h(prev, _h_new);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetRNNBuilderCellState(
    dynetRNNBuilder_t *builder, int32_t prev,
    const dynetExpression_t *const *c_new, size_t n) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(c_new);
  std::vector<dynet::Expression> _c_new;
  dynet_c::internal::copy_array_of_c_ptrs_to_vector(c_new, n, &_c_new);
  to_cpp_ptr(builder)->set_s(prev, _c_new);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetAddRNNBuilderInput(
    dynetRNNBuilder_t *builder, const dynetExpression_t *x,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      to_cpp_ptr(builder)->add_input(*to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetAddRNNBuilderInputToState(
    dynetRNNBuilder_t *builder, int32_t prev, const dynetExpression_t *x,
    dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      to_cpp_ptr(builder)->add_input(prev, *to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetRewindRNNBuilderOneStep(dynetRNNBuilder_t *builder) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  to_cpp_ptr(builder)->rewind_one_step();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetRNNBuilderParentStatePointer(
    const dynetRNNBuilder_t *builder, int32_t p, int32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(builder)->get_head(p);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetRNNBuilderDropout(
    dynetRNNBuilder_t *builder, float d) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  to_cpp_ptr(builder)->set_dropout(d);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetDisableRNNBuilderDropout(dynetRNNBuilder_t *builder) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  to_cpp_ptr(builder)->disable_dropout();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetRNNBuilderLastOutput(
    const dynetRNNBuilder_t *builder, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(to_cpp_ptr(builder)->back());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetRNNBuilderFinalHiddenState(
    const dynetRNNBuilder_t *builder, dynetExpression_t **newobj,
    size_t *size) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(size);
  std::vector<dynet::Expression> hs = to_cpp_ptr(builder)->final_h();
  dynet_c::internal::move_vector_to_array_of_c_ptrs(&hs, newobj, size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetRNNBuilderHiddenState(
    const dynetRNNBuilder_t *builder, int32_t i, dynetExpression_t **newobj,
    size_t *size) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(size);
  std::vector<dynet::Expression> hs = to_cpp_ptr(builder)->get_h(i);
  dynet_c::internal::move_vector_to_array_of_c_ptrs(&hs, newobj, size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetRNNBuilderFinalCellState(
    const dynetRNNBuilder_t *builder, dynetExpression_t **newobj,
    size_t *size) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(size);
  std::vector<dynet::Expression> cs = to_cpp_ptr(builder)->final_s();
  dynet_c::internal::move_vector_to_array_of_c_ptrs(&cs, newobj, size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetRNNBuilderCellState(
    const dynetRNNBuilder_t *builder, int32_t i, dynetExpression_t **newobj,
    size_t *size) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(size);
  std::vector<dynet::Expression> cs = to_cpp_ptr(builder)->get_s(i);
  dynet_c::internal::move_vector_to_array_of_c_ptrs(&cs, newobj, size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetRNNBuilderNumH0Components(
    const dynetRNNBuilder_t *builder, int32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(builder)->num_h0_components();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCopyRNNBuilderParameters(
    dynetRNNBuilder_t *builder, const dynetRNNBuilder_t *params) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(params);
  to_cpp_ptr(builder)->copy(*to_cpp_ptr(params));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetRNNBuilderParameterCollection(
    dynetRNNBuilder_t *builder, dynetParameterCollection_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(&to_cpp_ptr(builder)->get_parameter_collection());
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateSimpleRNNBuilder(
    uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, DYNET_C_BOOL support_lags,
    dynetRNNBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(model);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::SimpleRNNBuilder(
      layers, input_dim, hidden_dim, *to_cpp_ptr(model), support_lags));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetAddSimpleRNNBuilderAuxiliaryInput(
    dynetRNNBuilder_t *builder, const dynetExpression_t *x,
    const dynetExpression_t *aux, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(aux);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      reinterpret_cast<dynet::SimpleRNNBuilder*>(builder)
          ->add_auxiliary_input(*to_cpp_ptr(x), *to_cpp_ptr(aux)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetSimpleRNNBuilderDropout(
    dynetRNNBuilder_t *builder, float d, float d_h) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::SimpleRNNBuilder*>(builder)->set_dropout(d, d_h);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetSimpleRNNBuilderDropoutMasks(
    dynetRNNBuilder_t *builder, uint32_t batch_size) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::SimpleRNNBuilder*>(builder)
      ->set_dropout_masks(batch_size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateCoupledLSTMBuilder(
    uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, dynetRNNBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(model);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::CoupledLSTMBuilder(
      layers, input_dim, hidden_dim, *to_cpp_ptr(model)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetCoupledLSTMBuilderDropout(
    dynetRNNBuilder_t *builder, float d, float d_h, float d_c) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::CoupledLSTMBuilder*>(builder)
      ->set_dropout(d, d_h, d_c);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetCoupledLSTMBuilderDropoutMasks(
    dynetRNNBuilder_t *builder, uint32_t batch_size) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::CoupledLSTMBuilder*>(builder)
      ->set_dropout_masks(batch_size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateVanillaLSTMBuilder(
    uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, DYNET_C_BOOL ln_lstm, float forget_bias,
    dynetRNNBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(model);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::VanillaLSTMBuilder(
      layers, input_dim, hidden_dim, *to_cpp_ptr(model),
      ln_lstm, forget_bias));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetVanillaLSTMBuilderDropout(
    dynetRNNBuilder_t *builder, float d, float d_r) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::VanillaLSTMBuilder*>(builder)->set_dropout(d, d_r);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetVanillaLSTMBuilderDropoutMasks(
    dynetRNNBuilder_t *builder, uint32_t batch_size) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::VanillaLSTMBuilder*>(builder)
      ->set_dropout_masks(batch_size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateCompactVanillaLSTMBuilder(
    uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, dynetRNNBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(model);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::CompactVanillaLSTMBuilder(
      layers, input_dim, hidden_dim, *to_cpp_ptr(model)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetCompactVanillaLSTMBuilderDropout(
    dynetRNNBuilder_t *builder, float d, float d_r) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::CompactVanillaLSTMBuilder*>(builder)
      ->set_dropout(d, d_r);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetCompactVanillaLSTMBuilderDropoutMasks(
    dynetRNNBuilder_t *builder, uint32_t batch_size) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::CompactVanillaLSTMBuilder*>(builder)
      ->set_dropout_masks(batch_size);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetCompactVanillaLSTMBuilderWeightnoise(
    dynetRNNBuilder_t *builder, float std) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::CompactVanillaLSTMBuilder*>(builder)
      ->set_weightnoise(std);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateFastLSTMBuilder(
    uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, dynetRNNBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(model);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::FastLSTMBuilder(
      layers, input_dim, hidden_dim, *to_cpp_ptr(model)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetAddTreeLSTMBuilderInput(
    dynetRNNBuilder_t *builder, int32_t id, int32_t *children, size_t n,
    const dynetExpression_t *x, dynetExpression_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  DYNET_C_CHECK_NOT_NULL(children);
  DYNET_C_CHECK_NOT_NULL(x);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr_from_value(
      reinterpret_cast<dynet::TreeLSTMBuilder*>(builder)->add_input(
          id, std::vector<int32_t>(children, children + n), *to_cpp_ptr(x)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetTreeLSTMBuilderNumElements(
    dynetRNNBuilder_t *builder, int32_t num) try {
  DYNET_C_CHECK_NOT_NULL(builder);
  reinterpret_cast<dynet::TreeLSTMBuilder*>(builder)->set_num_elements(num);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateNaryTreeLSTMBuilder(
    uint32_t N, uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, dynetRNNBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(model);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::NaryTreeLSTMBuilder(
      N, layers, input_dim, hidden_dim, *to_cpp_ptr(model)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateUnidirectionalTreeLSTMBuilder(
    uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, dynetRNNBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(model);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::UnidirectionalTreeLSTMBuilder(
      layers, input_dim, hidden_dim, *to_cpp_ptr(model)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateBidirectionalTreeLSTMBuilder(
    uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, dynetRNNBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(model);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::BidirectionalTreeLSTMBuilder(
      layers, input_dim, hidden_dim, *to_cpp_ptr(model)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateGRUBuilder(
    uint32_t layers, uint32_t input_dim, uint32_t hidden_dim,
    dynetParameterCollection_t *model, dynetRNNBuilder_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(model);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::GRUBuilder(
      layers, input_dim, hidden_dim, *to_cpp_ptr(model)));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
