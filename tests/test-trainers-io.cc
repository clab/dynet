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

#define DYNET_TRAINER_IO_TEST_CASE(name, TRAINER_TYPE)          \
BOOST_AUTO_TEST_CASE(name) {                                    \
    /* build model */                                           \
    ParameterCollection m;                                      \
    TRAINER_TYPE trainer(m);                                    \
    auto p = build_pc(m);                                       \
    /* do one update and save params */                         \
    float loss1;                                                \
    loss1 = optimize(p, trainer);                               \
    {                                                           \
        dynet::TextFileSaver s("test.model");                   \
        s.save(m);                                              \
        std::ofstream f;                                        \
        f.open("test.optimizer");                               \
        trainer.save(f);                                        \
        f.close();                                              \
    }                                                           \
    /* do a second update and check that the loss where different */ \
    /* i.e. that the params of the model are actually different */ \
    auto loss2 = optimize(p, trainer);                          \
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
}                                                               \

#define DYNET_TRAINER_IO_PARAM_TEST_CASE(name, TRAINER_TYPE, SP_NAME) \
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
    Trainer trainer(m);                                    \
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

DYNET_TRAINER_IO_PARAM_TEST_CASE(momentum_sgd_io_param_vp, MomentumSGDTrainer, vp)
DYNET_TRAINER_IO_PARAM_TEST_CASE(momentum_sgd_io_param_vlp, MomentumSGDTrainer, vlp)

DYNET_TRAINER_IO_PARAM_TEST_CASE(adagrad_io_param_vp, AdagradTrainer, vp)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adagrad_io_param_vlp, AdagradTrainer, vlp)

DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_hg, AdadeltaTrainer, hg)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_hlg, AdadeltaTrainer, hlg)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_hd, AdadeltaTrainer, hd)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adadelta_io_param_hld, AdadeltaTrainer, hld)

DYNET_TRAINER_IO_PARAM_TEST_CASE(rmsprop_io_param_hmsg, RMSPropTrainer, hmsg)
DYNET_TRAINER_IO_PARAM_TEST_CASE(rmsprop_io_param_hlmsg, RMSPropTrainer, hlmsg)

DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_m, AdamTrainer, m)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_lm, AdamTrainer, lm)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_v, AdamTrainer, v)
DYNET_TRAINER_IO_PARAM_TEST_CASE(adam_io_param_lv, AdamTrainer, lv)

DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_m, AmsgradTrainer, m)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_lm, AmsgradTrainer, lm)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_v, AmsgradTrainer, v)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_lv, AmsgradTrainer, lv)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_vhat, AmsgradTrainer, vhat)
DYNET_TRAINER_IO_PARAM_TEST_CASE(amsgrad_io_param_lvhat, AmsgradTrainer, lvhat)

DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_hp, EGTrainer, hp)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_hlp, EGTrainer, hlp)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_zeg, EGTrainer, zeg)
DYNET_TRAINER_IO_PARAM_TEST_CASE(eg_io_param_meg, EGTrainer, meg)

DYNET_TRAINER_IO_TEST_CASE(simple_sgd_io, SimpleSGDTrainer)
DYNET_TRAINER_IO_TEST_CASE(cyclical_sgd_io, CyclicalSGDTrainer)
DYNET_TRAINER_IO_TEST_CASE(momentum_sgd_io, MomentumSGDTrainer)
DYNET_TRAINER_IO_TEST_CASE(adagrad_io, AdagradTrainer)
DYNET_TRAINER_IO_TEST_CASE(adadelta_io, AdadeltaTrainer)
DYNET_TRAINER_IO_TEST_CASE(rmsprop_io, RMSPropTrainer)
DYNET_TRAINER_IO_TEST_CASE(adam_io, AdamTrainer)
DYNET_TRAINER_IO_TEST_CASE(amsgrad_io, AmsgradTrainer)
//DYNET_TRAINER_IO_TEST_CASE(eg_io, EGTrainer)

BOOST_AUTO_TEST_SUITE_END()
