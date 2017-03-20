package edu.cmu.dynet

/** Interface for [[edu.cmu.dynet.Model]] trainers. You want to use a specific subclass. */
class Trainer private[dynet](_trainer: internal.Trainer) {
  def update(scale: Float = 1.0f): Unit = _trainer.update(scale)
  def updateEpoch(r: Float = 1.0f): Unit = _trainer.update_epoch(r)

  def clipGradients(scale: Float = 1.0f): Float = _trainer.clip_gradients(scale)
  def rescaleAndResetWeightDecay(): Unit = _trainer.rescale_and_reset_weight_decay()

  def status(): Unit = _trainer.status()

  def clippingEnabled: Boolean = _trainer.getClipping_enabled
  def clippingEnabled_=(b: Boolean) = _trainer.setClipping_enabled(b)

  def clipThreshold: Float = _trainer.getClip_threshold
  def clipThreshold_=(x: Float): Unit = _trainer.setClip_threshold(x)
}

class SimpleSGDTrainer private[dynet] (private[dynet] val trainer: internal.SimpleSGDTrainer)
  extends Trainer(trainer)
{
  def this(m: Model, e0: Float = 0.1f, edecay: Float = 0.0f) {
    this(new internal.SimpleSGDTrainer(m.model, e0, edecay))
  }
}

class MomentumSGDTrainer private[dynet] (private[dynet] val trainer: internal.MomentumSGDTrainer)
    extends Trainer(trainer)
{
  def this(m: Model, e0: Float = 0.1f, mom: Float = 0.9f, edecay: Float = 0.0f) {
    this(new internal.MomentumSGDTrainer(m.model, e0, mom, edecay))
  }
}

class AdagradTrainer private[dynet] (private[dynet] val trainer: internal.AdagradTrainer)
    extends Trainer(trainer)
{
  def this(m: Model, e0: Float = 0.1f, eps: Float = 1e-20f, edecay: Float = 0.0f) {
    this(new internal.AdagradTrainer(m.model, e0, eps, edecay))
  }
}

class RmsPropTrainer private[dynet] (private[dynet] val trainer: internal.RmsPropTrainer)
    extends Trainer(trainer)
{
  def this(m: Model, e0: Float = 0.1f, eps: Float = 1e-20f, rho: Float = 0.95f, edecay: Float = 0.0f) {
    this(new internal.RmsPropTrainer(m.model, e0, eps, rho, edecay))
  }
}

class AdamTrainer private[dynet] (private[dynet] val trainer: internal.AdamTrainer)
    extends Trainer(trainer)
{
  def this(m: Model, e0: Float = 0.001f, beta1: Float = 0.9f, beta2: Float = 0.999f,
    eps: Float = 1e-8f, edecay: Float = 0.0f) {
    this(new internal.AdamTrainer(m.model, e0, beta1, beta2, eps, edecay))
  }
}

