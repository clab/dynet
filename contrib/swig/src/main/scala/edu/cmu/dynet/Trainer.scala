package edu.cmu.dynet

/** Interface for [[edu.cmu.dynet.ParameterCollection]] trainers. You want to use a specific subclass. */
class Trainer private[dynet](_trainer: internal.Trainer) {
  def update(): Unit = _trainer.update()
  def updateEpoch(r: Float = 1.0f): Unit = _trainer.update_epoch(r)
  def restart(): Unit = _trainer.restart()
  def restart(lr: Float): Unit = _trainer.restart(lr)

  def clipGradients(): Float = _trainer.clip_gradients()
  def rescaleAndResetWeightDecay(): Unit = _trainer.rescale_and_reset_weight_decay()

  def status(): Unit = _trainer.status()

  def clippingEnabled: Boolean = _trainer.getClipping_enabled
  def clippingEnabled_=(b: Boolean) = _trainer.setClipping_enabled(b)

  def clipThreshold: Float = _trainer.getClip_threshold
  def clipThreshold_=(x: Float): Unit = _trainer.setClip_threshold(x)

  def learningRate:Float = _trainer.getLearning_rate()
  def learningRate_=(x:Float): Unit = _trainer.setLearning_rate(x)

  def enableSparseUpdates(): Unit = _trainer.setSparse_updates_enabled(true)
  def disableSparseUpdates(): Unit = _trainer.setSparse_updates_enabled(false)
  def isSparseUpdatesEnabled: Boolean = _trainer.getSparse_updates_enabled()
}

class SimpleSGDTrainer private[dynet] (private[dynet] val trainer: internal.SimpleSGDTrainer)
  extends Trainer(trainer)
{
  def this(m: ParameterCollection, learningRate: Float = 0.1f) {
    this(new internal.SimpleSGDTrainer(m.model, learningRate))
  }
}

class CyclicalSGDTrainer private[dynet] (private[dynet] val trainer: internal.CyclicalSGDTrainer)
  extends Trainer(trainer)
{
  def this(m: ParameterCollection, learningRateMin: Float = 0.01f, learningRateMax: Float = 0.1f, stepSize: Float = 2000f, gamma: Float = 0.0f, edecay: Float = 0.0f) {
    this(new internal.CyclicalSGDTrainer(m.model, learningRateMin, learningRateMax, stepSize, gamma, edecay))
  }
}

class MomentumSGDTrainer private[dynet] (private[dynet] val trainer: internal.MomentumSGDTrainer)
  extends Trainer(trainer)
{
  def this(m: ParameterCollection, learningRate: Float = 0.01f, mom: Float = 0.9f) {
    this(new internal.MomentumSGDTrainer(m.model, learningRate, mom))
  }
}

class AdagradTrainer private[dynet] (private[dynet] val trainer: internal.AdagradTrainer)
  extends Trainer(trainer)
{
  def this(m: ParameterCollection, learningRate: Float = 0.1f, eps: Float = 1e-20f) {
    this(new internal.AdagradTrainer(m.model, learningRate, eps))
  }
}

class AdadeltaTrainer private[dynet] (private[dynet] val trainer: internal.AdadeltaTrainer)
  extends Trainer(trainer)
{
  def this(m: ParameterCollection, eps: Float = 1e-6f, rho:Float = 0.95f) {
    this(new internal.AdadeltaTrainer(m.model, eps, rho))
  }
}

class RMSPropTrainer private[dynet] (private[dynet] val trainer: internal.RMSPropTrainer)
  extends Trainer(trainer)
{
  def this(m: ParameterCollection, learningRate: Float = 0.001f, eps: Float = 1e-8f, rho: Float = 0.9f) {
    this(new internal.RMSPropTrainer(m.model, learningRate, eps, rho))
  }
}

class AdamTrainer private[dynet] (private[dynet] val trainer: internal.AdamTrainer)
  extends Trainer(trainer)
{
  def this(m: ParameterCollection, learningRate: Float = 0.001f, beta1: Float = 0.9f, beta2: Float = 0.999f,
           eps: Float = 1e-8f) {
    this(new internal.AdamTrainer(m.model, learningRate, beta1, beta2, eps))
  }
}

class AmsgradTrainer private[dynet] (private[dynet] val trainer: internal.AmsgradTrainer)
  extends Trainer(trainer)
{
  def this(m: ParameterCollection, learningRate: Float = 0.001f, beta1: Float = 0.9f, beta2: Float = 0.999f,
           eps: Float = 1e-8f) {
    this(new internal.AmsgradTrainer(m.model, learningRate, beta1, beta2, eps))
  }
}

//class EGTrainer private[dynet] (private[dynet] val trainer: internal.EGTrainer) extends Trainer(trainer)
//{
//  def this(m: ParameterCollection, learningRate: Float = 0.1f, mom: Float = 0.9f, ne: Float = 0.0f) {
//    this(new internal.EGTrainer(m.model, learningRate, mom, ne))
//  }
//
//  def enableCyclicalLR(learningRateMin: Float = 0.01f, learningRateMax: Float = 0.1f, stepSize: Float = 2000f, gamma: Float = 0.0f) = {
//    trainer.enableCyclicalLR(learningRateMin, learningRateMax, stepSize, gamma)
//  }
//}
