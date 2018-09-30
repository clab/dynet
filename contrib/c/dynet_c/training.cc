#include <dynet_c/config.h>

#include <dynet/training.h>
#include <dynet_c/internal.h>
#include <dynet_c/training.h>

using dynet_c::internal::to_c_ptr;
using dynet_c::internal::to_cpp_ptr;

DYNET_C_STATUS dynetDeleteTrainer(dynetTrainer_t *trainer) try {
  DYNET_C_CHECK_NOT_NULL(trainer);
  delete to_cpp_ptr(trainer);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetUpdateTrainer(dynetTrainer_t *trainer) try {
  DYNET_C_CHECK_NOT_NULL(trainer);
  to_cpp_ptr(trainer)->update();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetRestartTrainer(dynetTrainer_t *trainer) try {
  DYNET_C_CHECK_NOT_NULL(trainer);
  to_cpp_ptr(trainer)->restart();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetRestartTrainerWithLearningRate(
    dynetTrainer_t *trainer, float lr) try {
  DYNET_C_CHECK_NOT_NULL(trainer);
  to_cpp_ptr(trainer)->restart(lr);
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetPrintTrainerStatus(dynetTrainer_t *trainer) try {
  DYNET_C_CHECK_NOT_NULL(trainer);
  to_cpp_ptr(trainer)->status();
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetTrainerLearningRate(
    const dynetTrainer_t *trainer, float *retval) try {
  DYNET_C_CHECK_NOT_NULL(trainer);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(trainer)->learning_rate;
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetGetTrainerClipThreshold(
    const dynetTrainer_t *trainer, float *retval) try {
  DYNET_C_CHECK_NOT_NULL(trainer);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(trainer)->clip_threshold;
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetSetTrainerClipThreshold(
    dynetTrainer_t *trainer, float threshold) try {
  DYNET_C_CHECK_NOT_NULL(trainer);
  dynet::Trainer *cpp_trainer = to_cpp_ptr(trainer);
  if (threshold <= 0.0) {
    cpp_trainer->clipping_enabled = false;
    cpp_trainer->clip_threshold = 0.0;
  } else {
    cpp_trainer->clipping_enabled = true;
    cpp_trainer->clip_threshold = threshold;
  }
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateSimpleSGDTrainer(
    dynetParameterCollection_t *m, float learning_rate,
    dynetTrainer_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(m);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj =
      to_c_ptr(new dynet::SimpleSGDTrainer(*to_cpp_ptr(m), learning_rate));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateCyclicalSGDTrainer(
    dynetParameterCollection_t *m, float learning_rate_min,
    float learning_rate_max, float step_size, float gamma,
    dynetTrainer_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(m);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::CyclicalSGDTrainer(
      *to_cpp_ptr(m), learning_rate_min, learning_rate_max, step_size, gamma));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateMomentumSGDTrainer(
    dynetParameterCollection_t *m, float learning_rate, float mom,
    dynetTrainer_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(m);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(
      new dynet::MomentumSGDTrainer(*to_cpp_ptr(m), learning_rate, mom));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateAdagradTrainer(
    dynetParameterCollection_t *m, float learning_rate, float eps,
    dynetTrainer_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(m);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(
      new dynet::AdagradTrainer(*to_cpp_ptr(m), learning_rate, eps));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateAdadeltaTrainer(
    dynetParameterCollection_t *m, float eps, float rho,
    dynetTrainer_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(m);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(
      new dynet::AdadeltaTrainer(*to_cpp_ptr(m), eps, rho));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateRMSPropTrainer(
    dynetParameterCollection_t *m, float learning_rate, float eps, float rho,
    dynetTrainer_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(m);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(
      new dynet::RMSPropTrainer(*to_cpp_ptr(m), learning_rate, eps, rho));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateAdamTrainer(
    dynetParameterCollection_t *m, float learning_rate, float beta_1,
    float beta_2, float eps, dynetTrainer_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(m);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::AdamTrainer(
      *to_cpp_ptr(m), learning_rate, beta_1, beta_2, eps));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS

DYNET_C_STATUS dynetCreateAmsgradTrainer(
    dynetParameterCollection_t *m, float learning_rate, float beta_1,
    float beta_2, float eps, dynetTrainer_t **newobj) try {
  DYNET_C_CHECK_NOT_NULL(m);
  DYNET_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new dynet::AmsgradTrainer(
      *to_cpp_ptr(m), learning_rate, beta_1, beta_2, eps));
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
