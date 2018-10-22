#define BOOST_TEST_MODULE TEST_TRAINER_IO
/*
 * There are two different macros to test an optimizer:
 * - DYNET_TRAINER_IO_TEST_CASE:
 *   makes two updates of a model. Parameters of the model and the optimizer
 *   are save after the first update. Then, a new model and a new optimizer
 *   are created and populated with the saved data.
 *   The two models must be equal after updating the second one.
 * - DYNET_TRAINER_IO_PARAM_TEST_CASE:
 *   this is a save/load test for a given attribute of the optimizer
 *
 * TODO: EGTrainer is no fully tested because the update does not change
 *       the model.
 */

#include <fstream>
#include <boost/test/unit_test.hpp>
#include <boost/version.hpp>

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/training.h>
#include <dynet/io.h>

#include "test.h"

using namespace dynet;
using namespace std;

struct TrainerIOTest {
  TrainerIOTest() {
    // initialize if necessary
    if(default_device == nullptr) {
      for (auto x : {"TrainerTest", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      ADD_EXTRA_ARGUMENTS(av)
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }
    ones_vals = {1.f,1.f,1.f};
    param_vals = {0.5f,0.1f,0.4f};
  }

  ~TrainerIOTest() {
    for (auto x : av) free(x);
  }

    Parameter build_pc(ParameterCollection& pc)
    {
        auto p = pc.add_parameters({3});
        TensorTools::set_elements(p.get_storage().values, param_vals);
        return p;
    }

    Expression build_loss(ComputationGraph& cg, Parameter& param)
    {
        auto x = parameter(cg, param);
        auto y = input(cg, {3}, ones_vals);
        return dot_product(x, y);
    }

    dynet::real optimize(Parameter& param, Trainer& trainer)
    {
        dynet::ComputationGraph cg;
        auto z = build_loss(cg, param);
        auto loss = as_scalar(cg.forward(z));
        cg.backward(z);
        trainer.update();

        return loss;
    }

  std::vector<float> ones_vals, param_vals;
  std::vector<char*> av;
};

BOOST_FIXTURE_TEST_SUITE(trainer_io_test, TrainerIOTest);

#define DYNET_TRAINER_IO_TEST_CASE(name, TRAINER_TYPE, MOVING_AVERAGE) \
BOOST_AUTO_TEST_CASE(name) {                                    \
    /* build model */                                           \
    ParameterCollection m;                                      \
    TRAINER_TYPE trainer1(m);                                   \
    if (MOVING_AVERAGE == MovingAverage::Exponential)                                                \
        trainer1.exponential_moving_average(0.999);                                    \
    if (MOVING_AVERAGE == MovingAverage::Cumulative)                                                \
        trainer1.cumulative_moving_average();                                    \
    auto p = build_pc(m);                                       \
    /* do one update and save params */                         \
    float loss1;                                                \
    loss1 = optimize(p, trainer1);                              \
    {                                                           \
        dynet::TextFileSaver s("test.model");                   \
        s.save(m);                                              \
        std::ofstream f;                                        \
        f.open("test.optimizer");                               \
        trainer1.save(f);                                       \
        f.close();                                              \
    }                                                           \
    /* do a second update and check that the loss where different */ \
    /* i.e. that the params of the model are actually different */ \
    auto loss2 = optimize(p, trainer1);                          \
    BOOST_CHECK(!(abs(loss1 - loss2)/abs(loss1) <= TOL && abs(loss1 - loss2)/abs(loss2) <= TOL)); \
    /*create a new model, load the save state+trainer params, */ \
    /* do one update and check that everything match */         \
    ParameterCollection m2;                                     \
    TRAINER_TYPE trainer2(m2);                                  \
    auto p2 = build_pc(m2);                                     \
    {                                                           \
        dynet::TextFileLoader s("test.model");                  \
        s.populate(m2);                                         \
        std::ifstream f;                                        \
        f.open("test.optimizer");                               \
        trainer2.populate(f);                                   \
        f.close();                                              \
    }                                                           \
    auto loss3 = optimize(p2, trainer2);                        \
    DYNET_CHECK_EQUAL(loss2, loss3);                            \
    DYNET_CHECK_EQUAL(m2, m);                                   \
    if (MOVING_AVERAGE != MovingAverage::None)                                                \
    {                                                           \
        trainer1.swap_params_to_moving_average();                  \
        trainer2.swap_params_to_moving_average();                  \
        DYNET_CHECK_EQUAL(m2, m);                               \
        trainer1.swap_params_to_weights();                      \
        trainer2.swap_params_to_weights();                      \
        if (MOVING_AVERAGE == MovingAverage::Exponential) \
        { \
        trainer1.swap_params_to_moving_average(true, true);              \
        trainer2.swap_params_to_moving_average(true, true);              \
        } \
        DYNET_CHECK_EQUAL(m2, m);                               \
    }                                                           \
}                                                               \

#define DYNET_TRAINER_IO_PARAM_TEST_CASE(name, TRAINER_TYPE, SP_NAME, MOVING_AVERAGE) \
BOOST_AUTO_TEST_CASE(name)                                      \
{                                                               \
    class Trainer : public TRAINER_TYPE                         \
    {                                                           \
        public:                                                 \
            using TRAINER_TYPE::SP_NAME;                        \
            explicit Trainer(ParameterCollection& m) : \
                TRAINER_TYPE::TRAINER_TYPE(m) \
                {} \
    };                                                          \
    /* build model */                                           \
    ParameterCollection m;                                      \
    Trainer trainer(m);                                         \
    if (MOVING_AVERAGE == MovingAverage::Exponential)                                                \
        trainer.exponential_moving_average(0.999);                                    \
    if (MOVING_AVERAGE == MovingAverage::Cumulative)                                                \
        trainer.cumulative_moving_average();                                    \
    auto p = build_pc(m);                                       \
    /* do one update and save params */                         \
    optimize(p, trainer);                                       \
    {                                                           \
        std::ofstream f;                                        \
        f.open("test.optimizer");                               \
        trainer.save(f);                                        \
        f.close();                                              \
    }                                                           \
    /*create a new model, load the save state params, */        \
    ParameterCollection m2;                                     \
    Trainer trainer2(m2);                                  \
    auto p2 = build_pc(m2);                                     \
    {                                                           \
        std::ifstream f;                                        \
        f.open("test.optimizer");                               \
        trainer2.populate(f);                                   \
        f.close();                                              \
    }                                                           \
    DYNET_CHECK_EQUAL(trainer2.SP_NAME, trainer.SP_NAME);                 \
}                                                               \

DYNET_TRAINER_IO_PARAM_TEST_CASE(momentum_sgd_io_param_vp, MomentumSGDTrainer, vp, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(momentum_sgd_io_param_vlp, MomentumSGDTrainer, vlp, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(momentum_sgd_io_param_ma_p_cumulative, MomentumSGDTrainer, ma_p, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(momentum_sgd_io_param_ma_p_exponential, MomentumSGDTrainer, ma_p, MovingAverage::Exponential)
DYNET_TRAINER_IO_PARAM_TEST_CASE(momentum_sgd_io_param_ma_lp_cumulative, MomentumSGDTrainer, ma_lp, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(momentum_sgd_io_param_ma_lp_exponential, MomentumSGDTrainer, ma_lp, MovingAverage::Exponential)

DYNET_TRAINER_IO_PARAM_TEST_CASE(adagrad_io_param_vp, AdagradTrainer, vp, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adagrad_io_param_vlp, AdagradTrainer, vlp, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adagrad_io_param_ma_p_cumulative, AdagradTrainer, ma_p, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adagrad_io_param_ma_p_exponential, AdagradTrainer, ma_p, MovingAverage::Exponential)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adagrad_io_param_ma_lp_cumulative, AdagradTrainer, ma_lp, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adagrad_io_param_ma_lp_exponential, AdagradTrainer, ma_lp, MovingAverage::Exponential)

DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_hg, AdadeltaTrainer, hg, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_hlg, AdadeltaTrainer, hlg, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_hd, AdadeltaTrainer, hd, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_hld, AdadeltaTrainer, hld, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_ma_p_cumulative, AdadeltaTrainer, ma_p, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_ma_p_exponential, AdadeltaTrainer, ma_p, MovingAverage::Exponential)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_ma_lp_cumulative, AdadeltaTrainer, ma_lp, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_ma_lp_exponential, AdadeltaTrainer, ma_lp, MovingAverage::Exponential)

DYNET_TRAINER_IO_PARAM_TEST_CASE(rmsprop_io_param_hmsg, RMSPropTrainer, hmsg, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(rmsprop_io_param_hlmsg, RMSPropTrainer, hlmsg, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(rmsprop_io_param_ma_p_cumulative, RMSPropTrainer, ma_p, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(rmsprop_io_param_ma_p_exponential, RMSPropTrainer, ma_p, MovingAverage::Exponential)
DYNET_TRAINER_IO_PARAM_TEST_CASE(rmsprop_io_param_ma_lp_cumulative, RMSPropTrainer, ma_lp, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(rmsprop_io_param_ma_lp_exponential, RMSPropTrainer, ma_lp, MovingAverage::Exponential)

DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_m, AdamTrainer, m, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_lm, AdamTrainer, lm, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_v, AdamTrainer, v, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_lv, AdamTrainer, lv, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_ma_p_cumulative, AdamTrainer, ma_p, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_ma_p_exponential, AdamTrainer, ma_p, MovingAverage::Exponential)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_ma_lp_cumulative, AdamTrainer, ma_lp, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_ma_lp_exponential, AdamTrainer, ma_lp, MovingAverage::Exponential)

DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_m, AmsgradTrainer, m, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_lm, AmsgradTrainer, lm, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_v, AmsgradTrainer, v, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_lv, AmsgradTrainer, lv, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_vhat, AmsgradTrainer, vhat, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_lvhat, AmsgradTrainer, lvhat, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_ma_p_cumulative, AmsgradTrainer, ma_p, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_ma_p_exponential, AmsgradTrainer, ma_p, MovingAverage::Exponential)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_ma_lp_cumulative, AmsgradTrainer, ma_lp, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_ma_lp_exponential, AmsgradTrainer, ma_lp, MovingAverage::Exponential)

DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_hp, EGTrainer, hp, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_hlp, EGTrainer, hlp, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_zeg, EGTrainer, zeg, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_meg, EGTrainer, meg, MovingAverage::None)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_ma_p_cumulative, EGTrainer, ma_p, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_ma_p_exponential, EGTrainer, ma_p, MovingAverage::Exponential)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_ma_lp_cumulative, EGTrainer, ma_lp, MovingAverage::Cumulative)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_ma_lp_exponential, EGTrainer, ma_lp, MovingAverage::Exponential)

DYNET_TRAINER_IO_TEST_CASE(simple_sgd_io, SimpleSGDTrainer, MovingAverage::None)
DYNET_TRAINER_IO_TEST_CASE(simple_sgd_io_ma_cumulative, SimpleSGDTrainer, MovingAverage::Cumulative)
DYNET_TRAINER_IO_TEST_CASE(simple_sgd_io_ma_exponential, SimpleSGDTrainer, MovingAverage::Exponential)
DYNET_TRAINER_IO_TEST_CASE(cyclical_sgd_io, CyclicalSGDTrainer, MovingAverage::None)
DYNET_TRAINER_IO_TEST_CASE(cyclical_sgd_io_ma_cumulative, CyclicalSGDTrainer, MovingAverage::Cumulative)
DYNET_TRAINER_IO_TEST_CASE(cyclical_sgd_io_ma_exponential, CyclicalSGDTrainer, MovingAverage::Exponential)
DYNET_TRAINER_IO_TEST_CASE(momentum_sgd_io, MomentumSGDTrainer, MovingAverage::None)
DYNET_TRAINER_IO_TEST_CASE(momentum_sgd_io_ma_cumulative, MomentumSGDTrainer, MovingAverage::Cumulative)
DYNET_TRAINER_IO_TEST_CASE(momentum_sgd_io_ma_exponential, MomentumSGDTrainer, MovingAverage::Exponential)
DYNET_TRAINER_IO_TEST_CASE(adagrad_io, AdagradTrainer, MovingAverage::None)
DYNET_TRAINER_IO_TEST_CASE(adagrad_io_ma_cumulative, AdagradTrainer, MovingAverage::Cumulative)
DYNET_TRAINER_IO_TEST_CASE(adagrad_io_ma_exponential, AdagradTrainer, MovingAverage::Exponential)
DYNET_TRAINER_IO_TEST_CASE(adadelta_io, AdadeltaTrainer, MovingAverage::None)
DYNET_TRAINER_IO_TEST_CASE(adadelta_io_ma_cumulative, AdadeltaTrainer, MovingAverage::Cumulative)
DYNET_TRAINER_IO_TEST_CASE(adadelta_io_ma_exponential, AdadeltaTrainer, MovingAverage::Exponential)
DYNET_TRAINER_IO_TEST_CASE(rmsprop_io, RMSPropTrainer, MovingAverage::None)
DYNET_TRAINER_IO_TEST_CASE(rmsprop_io_ma_cumulative, RMSPropTrainer, MovingAverage::Cumulative)
DYNET_TRAINER_IO_TEST_CASE(rmsprop_io_ma_exponential, RMSPropTrainer, MovingAverage::Exponential)
DYNET_TRAINER_IO_TEST_CASE(adam_io, AdamTrainer, MovingAverage::None)
DYNET_TRAINER_IO_TEST_CASE(adam_io_ma_cumulative, AdamTrainer, MovingAverage::Cumulative)
DYNET_TRAINER_IO_TEST_CASE(adam_io_ma_exponential, AdamTrainer, MovingAverage::Exponential)
DYNET_TRAINER_IO_TEST_CASE(amsgrad_io, AmsgradTrainer, MovingAverage::None)
DYNET_TRAINER_IO_TEST_CASE(amsgrad_io_ma_cumulative, AmsgradTrainer, MovingAverage::Cumulative)
DYNET_TRAINER_IO_TEST_CASE(amsgrad_io_ma_exponential, AmsgradTrainer, MovingAverage::Exponential)
//DYNET_TRAINER_IO_TEST_CASE(eg_io, EGTrainer, false)
//DYNET_TRAINER_IO_TEST_CASE(eg_io_ma, EGTrainer, true)

BOOST_AUTO_TEST_SUITE_END()
