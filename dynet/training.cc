#include "dynet/tensor-eigen.h"
#include "dynet/training.h"
#include "dynet/devices.h"

// #include "dynet/gpu-ops.h"
#include "dynet/param-nodes.h"
#include "dynet/weight-decay.h"
#include "dynet/io.h"

// same as in dynet/io.cc
static const int FLOAT32_PRECISION = 8;

// Macros for defining parameter update functions
#ifdef __CUDACC__
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real gscale, const std::vector<Tensor*> & values);
#elif defined(HAVE_CUDA)
// This is correct, but dying when models are read and written.
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  extern template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real gscale, const std::vector<Tensor*> & values); \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real gscale, const std::vector<Tensor*> & values) { \
    if(values[0]->device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)values[0]->device,gscale,values); } \
    else if(values[0]->device->type == DeviceType::GPU) { \
      cudaSetDevice(((Device_GPU*)values[0]->device)->cuda_device_id); \
      update_rule_dev(*(Device_GPU*)values[0]->device,gscale,values); } \
    else { throw std::runtime_error("Bad device in MyTrainer::update_rule"); } \
  }
#else
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real gscale, const std::vector<Tensor*> & values) { \
    if(values[0]->device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)values[0]->device,gscale,values); } \
    else { throw std::runtime_error("Bad device in MyTrainer::update_rule"); } \
  }
#endif

namespace dynet {

using namespace std;

template <class Derived>
bool is_valid(const Eigen::MatrixBase<Derived>& x) {
  return ((x - x).array() == (x - x).array()).all();
}

#ifndef __CUDACC__
namespace
{
// these functions are used as helper to write/read the optimizer state

void write_trainer_header(std::ostream& os, const std::string &type, const unsigned np, const unsigned nlp)
{
    // save information about this trainer status: name
    // + number of parameters
    // + number of lookup parameters
    os << type << ' ' << np << ' ' << nlp << std::endl;
}

void read_trainer_header(std::istream& is, const std::string& expected_type, unsigned* np, unsigned* nlp)
{
    std::string line, type;

    std::getline(is, line);
    std::istringstream iss(line);

    iss >> type >> *np >> *nlp;

    if (type != expected_type)
        DYNET_RUNTIME_ERR("Type does not match expected type")
}

void write_trainer_params(std::ostream& os, const std::vector<ShadowParameters>& vp)
{
    for (auto sp : vp)
         os 
            << "#Parameter# " << sp.h.d.size() << ' '
            << dynet::as_vector(sp.h)
            << std::endl
        ;
}

void write_trainer_params(std::ostream& os, const std::vector<ShadowLookupParameters>& vlp)
{
    for (auto slp : vlp)
         os 
            << "#LookupParameter# " << slp.all_h.d.size() << ' '
            << dynet::as_vector(slp.all_h)
            << std::endl
        ;
}

void read_trainer_params(std::istream& is, std::vector<ShadowParameters>& vp, const unsigned np)
{
    std::string line, type;
    unsigned s;
    std::vector<float> values;

    // load save params
    for (unsigned i = 0u ; i < np ; ++i)
    {
        auto& sp = vp[i];
        values.resize(sp.h.d.size());

        std::getline(is, line);
        std::istringstream iss(line);

        iss >> type >> s;
        if (type != "#Parameter#")
            DYNET_RUNTIME_ERR("Expected parameter");
        if (s != values.size())
            DYNET_RUNTIME_ERR("Dimension mismatch")
        iss >> values;

        TensorTools::set_elements(sp.h, values);
    }

    // empty extra params
    for (unsigned i = np ; i < vp.size() ; ++i)
        TensorTools::zero(vp[i].h);
}

void read_trainer_params(std::istream& is, std::vector<ShadowLookupParameters> vlp, const unsigned nlp)
{
    std::string line, type;
    unsigned s;
    std::vector<float> values;

    for (unsigned i = 0u ; i < nlp ; ++i)
    {
        auto& slp = vlp[i];
        values.resize(slp.all_h.d.size());

        std::getline(is, line);
        std::istringstream iss(line);

        iss >> type >> s;
        if (type != "#LookupParameter#")
            DYNET_RUNTIME_ERR("Expected parameter");
        if (s != values.size())
            DYNET_RUNTIME_ERR("Dimension mismatch")
        iss >> values;

        TensorTools::set_elements(slp.all_h, values);
    }

    // empty extra params
    for (unsigned i = nlp ; i < vlp.size() ; ++i)
        TensorTools::zero(vlp[i].all_h);
}

}
#endif

// --- The actual update code for each operation, implemented on various devices

// Trainer base class is run on CPUs
#ifndef __CUDACC__

Trainer::~Trainer() {}

void Trainer::rescale_and_reset_weight_decay() {
  const float weight_decay = model->get_weight_decay().current_weight_decay();
  for (auto p : model->parameters_list())
    if (p->is_updated())
      p->scale_parameters(weight_decay);
  for (auto p : model->lookup_parameters_list())
    if (p->is_updated())
      p->scale_parameters(weight_decay);
  model->get_weight_decay().reset_weight_decay();
}

float Trainer::clip_gradients() {
  float gscale = 1;
  if (clipping_enabled) {
    float gg = model->gradient_l2_norm();
    if (isnan(gg) || isinf(gg)) {
      ostringstream oss; oss << "Magnitude of gradient is bad: " << gg;
      throw std::runtime_error(oss.str());
    }
    if (gg > clip_threshold) {
      ++clips;
      ++clips_since_status;
      gscale = clip_threshold / gg;
    }
  }
  return gscale;
}

void Trainer::update_epoch(real r) {
  cerr << "Trainer::update_epoch has been deprecated and doesn't do anything. Please remove it from your code, and control the learning rate of the trainer directly, for example by: 'trainer.learning_rate /= (1 - rate_decay)', see https://github.com/clab/dynet/pull/695 for details." << endl;
}

// this calls the rule-specific updates over all updated parameters
void Trainer::update() {
  const auto & params = model->parameters_list();
  const auto & lparams = model->lookup_parameters_list();

  // Allocate if necessary
  if(aux_allocated < params.size()) {
    aux_allocated = alloc_impl();
  }
  if(aux_allocated_lookup < lparams.size()) {
    aux_allocated_lookup = alloc_lookup_impl();
  }

  // Perform gradient clipping and cycle through parameters
  const float gscale = clip_gradients();
  for(size_t i = 0; i < params.size(); ++i) {
    if(params[i]->updated) {
      update_params(gscale, i);
      params[i]->clear();
    }
  }
  for(size_t i = 0; i < lparams.size(); ++i) {
    auto &p = lparams[i];
    if (p->updated) {
      if(sparse_updates_enabled && !p->all_updated) {
        for (auto j : p->non_zero_grads)
          update_lookup_params(gscale, i, j);
      } else {
        update_lookup_params(gscale, i);
      }
      p->clear();
    }
  }

  // EMA
  if (moving_average() != MovingAverage::None && static_cast<unsigned int>(updates) % ma_update_freq == 0u)
  {
    if (ma_aux_allocated < params.size())
    {
        allocate_shadow_parameters(*model, ma_aux_allocated, ma_p);
        ma_aux_allocated = ma_p.size();
    }
    if (ma_aux_allocated_lookup < lparams.size())
    {
        allocate_shadow_lookup_parameters(*model, ma_aux_allocated_lookup, ma_lp);
        ma_aux_allocated_lookup = ma_lp.size();
    }

    swap_params_to_weights();
    for(size_t i = 0; i < params.size(); ++i)
    {
        Tensor& weights = params[i]->values;
        Tensor& ma = ma_p[i].h;
        update_ma_rule(&ma, &weights);
    }
    for(size_t i = 0; i < lparams.size(); ++i)
    {
        Tensor& weights = lparams[i]->all_values;
        Tensor& ma = ma_lp[i].all_h;
        update_ma_rule(&ma, &weights);
    }
    ++ ma_updates;
  }

  ++updates;
  ++updates_since_status;

  L2WeightDecay & wd = model->get_weight_decay();
  wd.update_weight_decay(); // update global weight scale
  if (wd.parameters_need_rescaled())
    rescale_and_reset_weight_decay();  // if wdscale is getting to small multiply all weights by wdscale, and set wdscale to 1
}

void Trainer::restart(real lr) {
    this->learning_rate = lr;
    this->restart();
}

void Trainer::save(std::ostream& os)
{
    os.precision(FLOAT32_PRECISION);
    os << std::scientific << std::showpos;
    write_trainer_header(os, "#Trainer#", aux_allocated, aux_allocated_lookup);
    os
        << learning_rate << ' '
        << clipping_enabled << ' '
        << clip_threshold << ' '
        << updates << ' '
        << ema_beta << ' '
        << ma_mode << ' '
        << ma_params_swapped << ' '
        << ma_params_saved << ' '
        << ma_update_freq << ' '
        << ma_updates 
        << std::endl
    ;


    // save EMA/CMA state if parameters are not swapped
    if (ma_mode != MovingAverage::None && !ma_params_swapped)
    {
        os << "[MA:TRUE]\n";
        write_trainer_header(os, "#MA#", ma_aux_allocated, ma_aux_allocated_lookup);
        write_trainer_params(os, ma_p);
        write_trainer_params(os, ma_lp);
    }
    else
    {
        os << "[MA:FALSE]\n";
    }
}

void Trainer::populate(std::istream& is)
{
    const auto& params = model->parameters_list();
    const auto& lparams = model->lookup_parameters_list();
    // Allocate if necessary
    if(aux_allocated < params.size())
        aux_allocated = alloc_impl();
    if(aux_allocated_lookup < lparams.size())
        aux_allocated_lookup = alloc_lookup_impl();

    unsigned np, nlp;
    read_trainer_header(is, "#Trainer#", &np, &nlp);

    if (np > params.size())
        DYNET_RUNTIME_ERR("Size mismatch")

    if (nlp > lparams.size())
        DYNET_RUNTIME_ERR("Size mismatch")

    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    iss
        >> learning_rate >> clipping_enabled >> clip_threshold >> updates
        >> ema_beta >> ma_mode >> ma_params_swapped >> ma_params_saved >> ma_update_freq >> ma_updates
    ;

    std::string ma_status;
    std::getline(is, ma_status);
    if (ma_status == "[MA:TRUE]")
    {
        if (ma_aux_allocated < params.size())
        {
            allocate_shadow_parameters(*model, ma_aux_allocated, ma_p);
            ma_aux_allocated = ma_p.size();
        }
        if (ma_aux_allocated_lookup < lparams.size())
        {
            allocate_shadow_lookup_parameters(*model, ma_aux_allocated_lookup, ma_lp);
            ma_aux_allocated_lookup = ma_lp.size();
        }

        unsigned ma_np, ma_nlp;
        read_trainer_header(is, "#MA#", &ma_np, &ma_nlp);
        if (ma_np > model->parameters_list().size())
            DYNET_RUNTIME_ERR("Size mismatch")

        if (ma_nlp > model->lookup_parameters_list().size())
            DYNET_RUNTIME_ERR("Size mismatch")

        read_trainer_params(is, ma_p, ma_np);
        read_trainer_params(is, ma_lp, ma_nlp);

    }
    else if (ma_status != "[MA:FALSE]")
    {
        DYNET_RUNTIME_ERR("Invalid moving averaged status");
    }
}

void Trainer::populate(std::istream& is, real lr)
{
    this->populate(is);
    this->learning_rate = lr;
}

#endif

// Moving Average

#ifdef __CUDACC__
    template void Trainer::update_ma_rule_dev<Device_GPU>(const Device_GPU& dev, Tensor* ma, Tensor* p);
    template void Trainer::swap_params_to_ma_rule_dev<Device_GPU>(const Device_GPU& dev, bool save_weights, bool bias_correction, Tensor* p, Tensor* mem, Tensor* ma);
    template void Trainer::swap_params_to_weights_rule_dev<Device_GPU>(const Device_GPU& dev, Tensor* p, Tensor* mem);
#elif defined(HAVE_CUDA)
    extern template void Trainer::update_ma_rule_dev<Device_GPU>(const Device_GPU& dev, Tensor* ma, Tensor* p);
    template void Trainer::update_ma_rule_dev<Device_CPU>(const Device_CPU& dev, Tensor* ma, Tensor* p);
    void Trainer::update_ma_rule(Tensor* ma, Tensor* p)
    {
        if(ma->device->type == DeviceType::CPU)
            update_ma_rule_dev(*(Device_CPU*)ma->device, ma, p);
        else if(ma->device->type == DeviceType::GPU)
        {
            cudaSetDevice(((Device_GPU*) ma->device)->cuda_device_id);
            update_ma_rule_dev(*(Device_GPU*) ma->device, ma, p);
        }
        else
            throw std::runtime_error("Bad device in MyTrainer::update_ma_rule");
    }

    extern template void Trainer::swap_params_to_ma_rule_dev<Device_GPU>(const Device_GPU& dev, bool save_weights, bool bias_correction, Tensor* p, Tensor* mem, Tensor* ma);
    template void Trainer::swap_params_to_ma_rule_dev<Device_CPU>(const Device_CPU& dev, bool save_weights, bool bias_correction, Tensor* p, Tensor* mem, Tensor* ma);
    void Trainer::swap_params_to_ma_rule(bool save_weights, bool bias_correction, Tensor* p, Tensor* mem, Tensor* ma)
    {
        if(ma->device->type == DeviceType::CPU)
            swap_params_to_ma_rule_dev(*(Device_CPU*)ma->device, save_weights, bias_correction, p, mem, ma);
        else if(ma->device->type == DeviceType::GPU)
        {
            cudaSetDevice(((Device_GPU*) ma->device)->cuda_device_id);
            swap_params_to_ma_rule_dev(*(Device_GPU*)ma->device, save_weights, bias_correction, p, mem, ma);
        }
        else
            throw std::runtime_error("Bad device in MyTrainer::swap_params_to_ma_rule");
    }


    extern template void Trainer::swap_params_to_weights_rule_dev<Device_GPU>(const Device_GPU& dev, Tensor* p, Tensor* mem);
    template void Trainer::swap_params_to_weights_rule_dev<Device_CPU>(const Device_CPU& dev, Tensor* p, Tensor* mem);
    void Trainer::swap_params_to_weights_rule(Tensor* p, Tensor* mem)
    {
        if(p->device->type == DeviceType::CPU)
            swap_params_to_weights_rule_dev(*(Device_CPU*)p->device, p, mem);
        else if(p->device->type == DeviceType::GPU)
        {
            cudaSetDevice(((Device_GPU*) p->device)->cuda_device_id);
            swap_params_to_weights_rule_dev(*(Device_GPU*)p->device, p, mem);
        }
        else
            throw std::runtime_error("Bad device in MyTrainer::swap_params_to_weights_rule");
    }
#else
    template void Trainer::update_ma_rule_dev<Device_CPU>(const Device_CPU& dev, Tensor* ma, Tensor* p);
    void Trainer::update_ma_rule(Tensor* ma, Tensor* p)
    {
        if(ma->device->type == DeviceType::CPU)
            update_ma_rule_dev(*(Device_CPU*) ma->device, ma, p);
        else
            throw std::runtime_error("Bad device in MyTrainer::update_ma_rule");
    }

    template void Trainer::swap_params_to_ma_rule_dev<Device_CPU>(const Device_CPU& dev, bool save_weights, bool bias_correction, Tensor* p, Tensor* mem, Tensor* ma);
    void Trainer::swap_params_to_ma_rule(bool save_weights, bool bias_correction, Tensor* p, Tensor* mem, Tensor* ma)
    {
        if(ma->device->type == DeviceType::CPU)
            swap_params_to_ma_rule_dev(*(Device_CPU*)ma->device, save_weights, bias_correction, p, mem, ma);
        else
            throw std::runtime_error("Bad device in MyTrainer::swap_params_to_ma_rule");
    }

    template void Trainer::swap_params_to_weights_rule_dev<Device_CPU>(const Device_CPU& dev, Tensor* p, Tensor* mem);
    void Trainer::swap_params_to_weights_rule(Tensor* p, Tensor* mem)
    {
        if(p->device->type == DeviceType::CPU)
            swap_params_to_weights_rule_dev(*(Device_CPU*) p->device, p, mem);
        else
            throw std::runtime_error("Bad device in MyTrainer::swap_params_to_weights_rule");
    }
#endif

template<class MyDevice>
void Trainer::update_ma_rule_dev(const MyDevice& dev, Tensor* ma, Tensor* p)
{
    switch (moving_average())
    {
        case MovingAverage::Cumulative:
            tvec(*ma).device(*dev.edevice) = (real(ma_updates) * tvec(*ma) + tvec(*p)) / (real(ma_updates)+1);
            break;
        case MovingAverage::Exponential:
            tvec(*ma).device(*dev.edevice) = ema_beta * tvec(*ma) + (1-ema_beta) * tvec(*p);
            break;
        case MovingAverage::None:
            // should not happen
            break;
    }
}

template<class MyDevice>
void Trainer::swap_params_to_ma_rule_dev(
    const MyDevice& dev,
    bool save_weights, bool bias_correction,
    Tensor* p, Tensor* mem, Tensor* ma
)
{
    if (save_weights)
        tvec(*mem).device(*dev.edevice) = tvec(*p);

    switch (moving_average())
    {
        case MovingAverage::Cumulative:
            tvec(*p).device(*dev.edevice) = tvec(*ma);
            break;
        case MovingAverage::Exponential:
            if (bias_correction)
            {
                const real pow_beta = pow(ema_beta, ma_updates);
                const real scale = 1.f / (1.f - pow_beta);
                tvec(*p).device(*dev.edevice) = scale * tvec(*ma);
            }
            else
                tvec(*p).device(*dev.edevice) = tvec(*ma);
            break;
        case MovingAverage::None:
            // should not happen
            break;
    }
}

template<class MyDevice>
void Trainer::swap_params_to_weights_rule_dev(const MyDevice& dev, Tensor* p, Tensor* mem)
{
    tvec(*p).device(*dev.edevice) = tvec(*mem);
}

#ifndef __CUDACC__

MovingAverage Trainer::moving_average()
{
    return ma_mode;
}

void Trainer::exponential_moving_average(float beta, unsigned update_freq)
{
    if (updates > 0)
        DYNET_RUNTIME_ERR("This function must be called before any update");
    if (update_freq == 0u)
        DYNET_RUNTIME_ERR("The update frequency cannot be null");

    ema_beta = beta;
    ma_update_freq = update_freq;
    ma_mode = MovingAverage::Exponential;
}

void Trainer::cumulative_moving_average(unsigned update_freq)
{
    if (updates > 0)
        DYNET_RUNTIME_ERR("This function must be called before any update");
    if (update_freq == 0u)
        DYNET_RUNTIME_ERR("The update frequency cannot be null");

    ma_update_freq = update_freq;
    ma_mode = MovingAverage::Cumulative;
}

void Trainer::swap_params_to_moving_average(bool save_weights, bool bias_correction)
{
    if (moving_average() == MovingAverage::None)
        DYNET_RUNTIME_ERR("Moving average is not enabled");
    if (ma_updates == 0u)
        DYNET_RUNTIME_ERR("Moving average has not been set yet");

    if (ma_params_swapped)
        return; // nothing to do
    ma_params_swapped = true;
    ma_params_saved = save_weights;

    const auto& params = model->parameters_list();
    const auto& lparams = model->lookup_parameters_list();

    // check memory (shadow params are automatically initialized to zero)
    if (ma_p.size() < params.size())
        allocate_shadow_parameters(*model, ma_p.size(), ma_p);
    if (ma_lp.size() < lparams.size())
        allocate_shadow_lookup_parameters(*model, ma_lp.size(), ma_lp);

    if (save_weights)
    {
        if (ma_saved_p.size() < params.size())
            allocate_shadow_parameters(*model, ma_saved_p.size(), ma_saved_p);
        if (ma_saved_lp.size() < lparams.size())
            allocate_shadow_lookup_parameters(*model, ma_saved_lp.size(), ma_saved_lp);
    }

    for(size_t i = 0; i < params.size(); ++i)
    {
        Tensor& weights = params[i]->values;
        Tensor& mem = ma_saved_p[i].h;
        Tensor& ma = ma_p[i].h;

        swap_params_to_ma_rule(save_weights, bias_correction, &weights, &mem, &ma);
    }
    for(size_t i = 0; i < lparams.size(); ++i)
    {
        Tensor& weights = lparams[i]->all_values;
        Tensor& mem = ma_saved_lp[i].all_h;
        Tensor& ma = ma_lp[i].all_h;

        swap_params_to_ma_rule(save_weights, bias_correction, &weights, &mem, &ma);
    }
}

void Trainer::swap_params_to_weights()
{
    if (!ma_params_swapped)
        return;
    if (!ma_params_saved)
        DYNET_RUNTIME_ERR("Weights have not been save.")
    ma_params_swapped = false;

    const auto& params = model->parameters_list();
    const auto& lparams = model->lookup_parameters_list();

    // if the number of the parameters has changed,
    // they are ignored.
    // Setting them to 0 would not be a good strategy
    // (i.e. we want to keep their init value, e.g. glorot)
    for(size_t i = 0; i < ma_saved_p.size(); ++i)
    {
        Tensor& weights = params[i]->values;
        Tensor& mem = ma_saved_p[i].h;
        swap_params_to_weights_rule(&weights, &mem);
    }
    for(size_t i = 0; i < ma_saved_lp.size(); ++i)
    {
        Tensor& weights = lparams[i]->all_values;
        Tensor& mem = ma_saved_lp[i].all_h;
        swap_params_to_weights_rule(&weights, &mem);
    }
}
#endif

// --- SimpleSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void SimpleSGDTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[0]).device(*dev.edevice) -= tvec(*ts[1]) * (learning_rate * gscale / model->get_weight_decay().current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(SimpleSGDTrainer)

#ifndef __CUDACC__
void SimpleSGDTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g});
}
void SimpleSGDTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx]});
}
void SimpleSGDTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads});
}
#endif

// --- CyclicalSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void CyclicalSGDTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[0]).device(*dev.edevice) -= tvec(*ts[1]) * (learning_rate * gscale / model->get_weight_decay().current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(CyclicalSGDTrainer)

#ifndef __CUDACC__
void CyclicalSGDTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g});
}
void CyclicalSGDTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx]});
}
void CyclicalSGDTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads});
}

#endif

// --- MomentumSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=momentum
template <class MyDevice>
void MomentumSGDTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * momentum - tvec(*ts[1]) * (learning_rate * gscale);
  tvec(*ts[0]).device(*dev.edevice) += tvec(*ts[2]) / model->get_weight_decay().current_weight_decay();
}
DYNET_TRAINER_INST_DEV_IMPL(MomentumSGDTrainer)

#ifndef __CUDACC__
void MomentumSGDTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &vp[idx].h});
}
void MomentumSGDTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &vlp[idx].h[lidx]});
}
void MomentumSGDTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &vlp[idx].all_h});
}
unsigned MomentumSGDTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, vp);
  return vp.size();
}
unsigned MomentumSGDTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, vlp);
  return vlp.size();
}

void MomentumSGDTrainer::restart() {
  for (auto sp : vp)
    TensorTools::zero(sp.h);
  for (auto slp : vlp)
    TensorTools::zero(slp.all_h);
}

void MomentumSGDTrainer::save(std::ostream& os)
{
    Trainer::save(os);

    write_trainer_header(os, "#MomentumSGDTrainer#", aux_allocated, aux_allocated_lookup);
    write_trainer_params(os, vp);
    write_trainer_params(os, vlp);
    os << momentum << std::endl;
}

void MomentumSGDTrainer::populate(std::istream& is)
{
    Trainer::populate(is);

    unsigned np, nlp;
    read_trainer_header(is, "#MomentumSGDTrainer#", &np, &nlp);
    read_trainer_params(is, vp, np);
    read_trainer_params(is, vlp, nlp);

    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    iss >> momentum;
}

#endif

// --- AdagradTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=stddev
template <class MyDevice>
void AdagradTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale;
  tvec(*ts[2]).device(*dev.edevice) += tvec(*ts[1]).square();
  tvec(*ts[0]).device(*dev.edevice) += tvec(*ts[1]) / (tvec(*ts[2]) + epsilon).sqrt() * (-learning_rate / model->get_weight_decay().current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(AdagradTrainer)

#ifndef __CUDACC__
void AdagradTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &vp[idx].h});
}
void AdagradTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &vlp[idx].h[lidx]});
}
void AdagradTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &vlp[idx].all_h});
}
unsigned AdagradTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, vp);
  return vp.size();
}
unsigned AdagradTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, vlp);
  return vlp.size();
}

void AdagradTrainer::restart() {
  for (auto sp : vp)
    TensorTools::zero(sp.h);
  for (auto slp : vlp)
    TensorTools::zero(slp.all_h);
}

void AdagradTrainer::save(std::ostream& os)
{
    Trainer::save(os);

    write_trainer_header(os, "#AdagradTrainer#", aux_allocated, aux_allocated_lookup);
    write_trainer_params(os, vp);
    write_trainer_params(os, vlp);
    os << epsilon<< std::endl;
}

void AdagradTrainer::populate(std::istream& is)
{
    Trainer::populate(is);

    unsigned np, nlp;
    read_trainer_header(is, "#AdagradTrainer#", &np, &nlp);
    read_trainer_params(is, vp, np);
    read_trainer_params(is, vlp, nlp);

    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    iss >> epsilon;
}

#endif

// --- AdadeltaTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=hg, ts[3]=hd
template <class MyDevice>
void AdadeltaTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale;
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * rho + tvec(*ts[1]).square() * (1.f - rho);
  tvec(*ts[1]).device(*dev.edevice) = - tvec(*ts[1]) * (tvec(*ts[3]) + epsilon).sqrt() / (tvec(*ts[2]) + epsilon).sqrt();
  tvec(*ts[3]).device(*dev.edevice) = tvec(*ts[3]) * rho + tvec(*ts[1]).square() * (1.f - rho);
  tvec(*ts[0]).device(*dev.edevice) += tvec(*ts[1]) / model->get_weight_decay().current_weight_decay();
}
DYNET_TRAINER_INST_DEV_IMPL(AdadeltaTrainer)

#ifndef __CUDACC__
void AdadeltaTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &hg[idx].h, &hd[idx].h});
}
void AdadeltaTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &hlg[idx].h[lidx], &hld[idx].h[lidx]});
}
void AdadeltaTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &hlg[idx].all_h, &hld[idx].all_h});
}
unsigned AdadeltaTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, hg);
  allocate_shadow_parameters(*model, aux_allocated, hd);
  return hd.size();
}
unsigned AdadeltaTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, hlg);
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, hld);
  return hld.size();
}

void AdadeltaTrainer::restart() {
  for (auto sp : hg)
    TensorTools::zero(sp.h);
  for (auto sp : hd)
    TensorTools::zero(sp.h);
  for (auto slp : hlg)
    TensorTools::zero(slp.all_h);
  for (auto slp : hld)
    TensorTools::zero(slp.all_h);
}

void AdadeltaTrainer::save(std::ostream& os)
{
    Trainer::save(os);

    write_trainer_header(os, "#AdadeltaTrainer#", aux_allocated, aux_allocated_lookup);
    write_trainer_params(os, hg);
    write_trainer_params(os, hd);
    write_trainer_params(os, hlg);
    write_trainer_params(os, hld);
    os << epsilon << ' ' << rho << std::endl;
}

void AdadeltaTrainer::populate(std::istream& is)
{
    Trainer::populate(is);

    unsigned np, nlp;
    read_trainer_header(is, "#AdadeltaTrainer#", &np, &nlp);
    read_trainer_params(is, hg, np);
    read_trainer_params(is, hd, np);
    read_trainer_params(is, hlg, nlp);
    read_trainer_params(is, hld, nlp);

    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    iss >> epsilon >> rho;
}

#endif

// --- RMSPropTrainer
// TODO: This is not finished yet, because it memorizes a scalar for each set of parameters, not each parameter itself.
//       We could implement this with one tensor for each scalar, but this is pretty wasteful

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void RMSPropTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale; // Scale gradient
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * rho + tvec(*ts[1]).square() * (1.f - rho); // Update square gradient exponential average
  tvec(*ts[1]).device(*dev.edevice) = - tvec(*ts[1]) / (tvec(*ts[2]) + epsilon).sqrt(); // Divide by the RMS
  tvec(*ts[0]).device(*dev.edevice) += learning_rate * tvec(*ts[1]) / model->get_weight_decay().current_weight_decay(); // Apply weight decay (should we do this?)
  // real& d2 = hg[pi++];
  // real g2 = p->vec(g).squaredNorm();
  // d2 = rho * d2 + (1.f - rho) * g2;
  // p->vec(values) -= ((learning_rate * gscale / sqrt(d2 + epsilon)) * p->vec(g)) / model->get_weight_decay().current_weight_decay();
}
DYNET_TRAINER_INST_DEV_IMPL(RMSPropTrainer)

#ifndef __CUDACC__
void RMSPropTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &hmsg[idx].h});
}
void RMSPropTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &hlmsg[idx].h[lidx]});
}
void RMSPropTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &hlmsg[idx].all_h});
}
unsigned RMSPropTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, hmsg);
  return hmsg.size();
}
unsigned RMSPropTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, hlmsg);
  return hlmsg.size();
}

void RMSPropTrainer::restart() {
  for (auto sp : hmsg)
    TensorTools::zero(sp.h);
  for (auto slp : hlmsg)
    TensorTools::zero(slp.all_h);
}

void RMSPropTrainer::save(std::ostream& os)
{
    Trainer::save(os);

    write_trainer_header(os, "#RMSPropTrainer#", aux_allocated, aux_allocated_lookup);
    write_trainer_params(os, hmsg);
    write_trainer_params(os, hlmsg);
    os << epsilon << ' ' << rho << std::endl;
}

void RMSPropTrainer::populate(std::istream& is)
{
    Trainer::populate(is);

    unsigned np, nlp;
    read_trainer_header(is, "#RMSPropTrainer#", &np, &nlp);
    read_trainer_params(is, hmsg, np);
    read_trainer_params(is, hlmsg, nlp);

    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    iss >> epsilon >> rho;
}

#endif

// --- AdamTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=mean, ts[3]=variance
template <class MyDevice>
void AdamTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale;
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * beta_1 + tvec(*ts[1]) * (1.f - beta_1);
  tvec(*ts[3]).device(*dev.edevice) = tvec(*ts[3]) * beta_2 + tvec(*ts[1]).square() * (1.f - beta_2);
  float lr_t = learning_rate * sqrt(1-pow(beta_2, updates+1))/(1-pow(beta_1, updates+1))/ model->get_weight_decay().current_weight_decay();
  tvec(*ts[0]).device(*dev.edevice) -= tvec(*ts[2]) / (tvec(*ts[3]).sqrt() + epsilon) * lr_t;
}
DYNET_TRAINER_INST_DEV_IMPL(AdamTrainer)

#ifndef __CUDACC__
void AdamTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &m[idx].h, &v[idx].h});
}
void AdamTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &lm[idx].h[lidx], &lv[idx].h[lidx]});
}
void AdamTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &lm[idx].all_h, &lv[idx].all_h});
}
unsigned AdamTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, m);
  allocate_shadow_parameters(*model, aux_allocated, v);
  return v.size();
}
unsigned AdamTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lm);
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lv);
  return lv.size();
}

void AdamTrainer::restart() {
  for (auto sp : m)
    TensorTools::zero(sp.h);
  for (auto sp : v)
    TensorTools::zero(sp.h);
  for (auto slp : lm)
    TensorTools::zero(slp.all_h);
  for (auto slp : lv)
    TensorTools::zero(slp.all_h);
}

void AdamTrainer::save(std::ostream& os)
{
    Trainer::save(os);

    write_trainer_header(os, "#AdamTrainer#", aux_allocated, aux_allocated_lookup);
    write_trainer_params(os, m);
    write_trainer_params(os, v);
    write_trainer_params(os, lm);
    write_trainer_params(os, lv);
    os << beta_1 << ' ' << beta_2 << ' ' << epsilon << std::endl;
}

void AdamTrainer::populate(std::istream& is)
{
    Trainer::populate(is);

    unsigned np, nlp;
    read_trainer_header(is, "#AdamTrainer#", &np, &nlp);
    read_trainer_params(is, m, np);
    read_trainer_params(is, v, np);
    read_trainer_params(is, lm, nlp);
    read_trainer_params(is, lv, nlp);

    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    iss >> beta_1 >> beta_2 >> epsilon;
}

#endif

// --- AMSGradTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=mean, ts[3]=variance, t[4]=max
template <class MyDevice>
void AmsgradTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  tvec(*ts[1]).device(*dev.edevice) = tvec(*ts[1]) * gscale;
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * beta_1 + tvec(*ts[1]) * (1.f - beta_1);
  tvec(*ts[3]).device(*dev.edevice) = tvec(*ts[3]) * beta_2 + tvec(*ts[1]).square() * (1.f - beta_2);
  tvec(*ts[4]).device(*dev.edevice) = tvec(*ts[4]).cwiseMax(tvec(*ts[3]));
  float lr_t = learning_rate * sqrt(1-pow(beta_2, updates+1))/(1-pow(beta_1, updates+1))/ model->get_weight_decay().current_weight_decay();
  tvec(*ts[0]).device(*dev.edevice) -= tvec(*ts[2]) / (tvec(*ts[4]).sqrt() + epsilon) * lr_t;
}
DYNET_TRAINER_INST_DEV_IMPL(AmsgradTrainer)

#ifndef __CUDACC__
void AmsgradTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &m[idx].h, &v[idx].h, &vhat[idx].h});
}
void AmsgradTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &lm[idx].h[lidx], &lv[idx].h[lidx], &lvhat[idx].h[lidx]});
}
void AmsgradTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_values, &p->all_grads, &lm[idx].all_h, &lv[idx].all_h, &lvhat[idx].all_h});
}
unsigned AmsgradTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, m);
  allocate_shadow_parameters(*model, aux_allocated, v);
  allocate_shadow_parameters(*model, aux_allocated, vhat);
  return vhat.size();
}
unsigned AmsgradTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lm);
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lv);
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, lvhat);
  return lvhat.size();
}

void AmsgradTrainer::restart() {
  for (auto sp : m)
    TensorTools::zero(sp.h);
  for (auto sp : v)
    TensorTools::zero(sp.h);
  for (auto sp : vhat)
    TensorTools::zero(sp.h);
  for (auto slp : lm)
    TensorTools::zero(slp.all_h);
  for (auto slp : lv)
    TensorTools::zero(slp.all_h);
  for (auto slp : lvhat)
    TensorTools::zero(slp.all_h);
}

void AmsgradTrainer::save(std::ostream& os)
{
    Trainer::save(os);

    write_trainer_header(os, "#AmsgradTrainer#", aux_allocated, aux_allocated_lookup);
    write_trainer_params(os, m);
    write_trainer_params(os, v);
    write_trainer_params(os, vhat);
    write_trainer_params(os, lm);
    write_trainer_params(os, lv);
    write_trainer_params(os, lvhat);
    os << beta_1 << ' ' << beta_2 << ' ' << epsilon << std::endl;
}

void AmsgradTrainer::populate(std::istream& is)
{
    Trainer::populate(is);

    unsigned np, nlp;
    read_trainer_header(is, "#AmsgradTrainer#", &np, &nlp);
    read_trainer_params(is, m, np);
    read_trainer_params(is, v, np);
    read_trainer_params(is, vhat, np);
    read_trainer_params(is, lm, nlp);
    read_trainer_params(is, lvhat, nlp);

    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    iss >> beta_1 >> beta_2 >> epsilon;
}

#endif

template <class MyDevice>
void EGTrainer::update_rule_dev(const MyDevice & dev, real gscale, const std::vector<Tensor*> & ts) {
  // Add momentum
  tvec(*ts[2]).device(*dev.edevice) = tvec(*ts[2]) * momentum - tvec(*ts[1]) * (learning_rate * gscale);
  tvec(*ts[0]).device(*dev.edevice) = tvec(*ts[0]).log() + tvec(*ts[2]) / model->get_weight_decay().current_weight_decay();// with momentum only
  TensorTools::logsumexp_dev(dev, *ts[0], *ts[3], *ts[4]);// z refers to logZ
  tvec(*ts[0]).device(*dev.edevice) = (tvec(*ts[0]) - as_scalar(*ts[4])).exp();// FIXME: other way(s) of not using as_scalar(z)?
}
DYNET_TRAINER_INST_DEV_IMPL(EGTrainer)

#ifndef __CUDACC__
// --- EGTrainer
EGTrainer::EGTrainer(ParameterCollection& mod, real learning_rate, real mom, real ne)
  : Trainer(mod, learning_rate), momentum(mom), isCyclical(false) {
  zeg.d = meg.d = {1};
  zeg.device = meg.device = default_device;
  default_device->allocate_tensor(DeviceMempool::PS, zeg);
  default_device->allocate_tensor(DeviceMempool::PS, meg);
}
void EGTrainer::update_params(real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(gscale, {&p->values, &p->g, &hp[idx].h, &meg, &zeg});
}
void EGTrainer::update_lookup_params(real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->values[lidx], &p->grads[lidx], &hlp[idx].h[lidx], &meg, &zeg});
}
void EGTrainer::update_lookup_params(real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(gscale, {&p->all_grads, &p->all_grads, &hlp[idx].all_h, &meg, &zeg});
}
unsigned EGTrainer::alloc_impl() {
  allocate_shadow_parameters(*model, aux_allocated, hp);
  return hp.size();
}
unsigned EGTrainer::alloc_lookup_impl() {
  allocate_shadow_lookup_parameters(*model, aux_allocated_lookup, hlp);
  return hlp.size();
}

void EGTrainer::restart() {
  for (auto sp : hp)
    TensorTools::zero(sp.h);
  for (auto slp : hlp)
    TensorTools::zero(slp.all_h);
}

void EGTrainer::save(std::ostream& os)
{
    Trainer::save(os);

    write_trainer_header(os, "#EGTrainer#", aux_allocated, aux_allocated_lookup);
    write_trainer_params(os, hp);
    write_trainer_params(os, hlp);
    float f_zeg = as_scalar(zeg);
    float f_meg = as_scalar(meg);
    os
        << f_zeg << ' '
        << f_meg << ' '
        << momentum << ' '
        << e_min << ' '
        << e_max << ' '
        << step_size << ' '
        << gamma << ' '
        << it << ' '
        << isCyclical
    ;
}

void EGTrainer::populate(std::istream& is)
{
    Trainer::populate(is);

    unsigned np, nlp;
    read_trainer_header(is, "#EGTrainer#", &np, &nlp);
    read_trainer_params(is, hp, np);
    read_trainer_params(is, hlp, nlp);

    std::string line;
    std::getline(is, line);
    std::istringstream iss(line);
    float f_zeg, f_meg;
    iss >> f_zeg >> f_meg >> momentum >> e_min >> e_max >> step_size >> gamma >> it >> isCyclical;
    TensorTools::set_element(zeg, 0u, f_zeg);
    TensorTools::set_element(meg, 0u, f_meg);
}


ostream& operator<<(std::ostream& os, const MovingAverage& o)
{
    switch (o)
    {
        case MovingAverage::None:
            os << "None";
            break;
        case MovingAverage::Cumulative:
            os << "Cumulative";
            break;
        case MovingAverage::Exponential:
            os << "Exponential";
            break;
    }
    return os;
}
istream& operator>>(std::istream& is, MovingAverage& o)
{
    std::string v;
    is >> v;
    if (v == "None")
        o = MovingAverage::None;
    else if (v == "Cumulative")
        o = MovingAverage::Cumulative;
    else if (v == "Exponential")
        o = MovingAverage::Exponential;
    else
        DYNET_RUNTIME_ERR("Invalid moving average mode: " << v);
    return is;
}

#endif


} // namespace dynet
