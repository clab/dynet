use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, ParameterCollection, Result, Wrap};

/// `Trainer` trait
pub trait Trainer: Wrap<dynet_sys::dynetTrainer_t> {
    /// Updates parameters.
    fn update(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetUpdateTrainer(self.as_mut_ptr()));
        }
    }

    /// Restarts the trainer.
    fn restart(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetRestartTrainer(self.as_mut_ptr()));
        }
    }

    /// Restarts the trainer with a new learning rate.
    fn restart_with_learning_rate(&mut self, lr: f32) {
        unsafe {
            check_api_status!(dynet_sys::dynetRestartTrainerWithLearningRate(
                self.as_mut_ptr(),
                lr
            ));
        }
    }

    /// Prints information about the trainer.
    fn print_status(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetPrintTrainerStatus(self.as_mut_ptr()));
        }
    }

    /// Gets global learning rate for all parameters.
    fn learning_rate(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(dynet_sys::dynetGetTrainerLearningRate(
                self.as_ptr(),
                &mut retval,
            ));
            retval
        }
    }

    /// Gets clipping threshold.
    fn clip_threshold(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(dynet_sys::dynetGetTrainerClipThreshold(
                self.as_ptr(),
                &mut retval,
            ));
            retval
        }
    }

    /// Sets clipping threshold.
    fn set_clip_threshold(&mut self, threshold: f32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetTrainerClipThreshold(
                self.as_mut_ptr(),
                threshold,
            ));
        }
    }
}

macro_rules! impl_trainer {
    ($name:ident) => {
        impl_wrap_owned!($name, dynetTrainer_t);
        impl_drop!($name, dynetDeleteTrainer);
        impl Trainer for $name {}
    };
}

/// A stochastic gradient descent trainer.
#[derive(Debug)]
pub struct SimpleSGDTrainer {
    inner: NonNull<dynet_sys::dynetTrainer_t>,
}

impl_trainer!(SimpleSGDTrainer);

impl SimpleSGDTrainer {
    /// Creates a new `SimpleSGDTrainer`.
    ///
    /// # Arguments
    ///
    /// * m - ParameterCollection to be trained.
    /// * learning_rate - Initial learning rate.
    pub fn new(m: &mut ParameterCollection, learning_rate: f32) -> SimpleSGDTrainer {
        unsafe {
            let mut trainer_ptr: *mut dynet_sys::dynetTrainer_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateSimpleSGDTrainer(
                m.as_mut_ptr(),
                learning_rate,
                &mut trainer_ptr
            ));
            SimpleSGDTrainer::from_raw(trainer_ptr, true)
        }
    }

    /// Creates a new `SimpleSGDTrainer` with default settings.
    ///
    /// This is equivalent to `SimpleSGDTrainer::new(m, 0.1)`.
    pub fn default(m: &mut ParameterCollection) -> SimpleSGDTrainer {
        SimpleSGDTrainer::new(m, 0.1)
    }
}

/// Cyclical learning rate SGD.
#[derive(Debug)]
pub struct CyclicalSGDTrainer {
    inner: NonNull<dynet_sys::dynetTrainer_t>,
}

impl_trainer!(CyclicalSGDTrainer);

impl CyclicalSGDTrainer {
    /// Creates a new `CyclicalSGDTrainer`.
    ///
    /// # Arguments
    ///
    /// * m ParameterCollection to be trained.
    /// * learning_rate_min Lower learning rate.
    /// * learning_rate_max Upper learning rate.
    /// * step_size Period of the triangular function in number of iterations (__not__ epochs).
    /// * gamma Learning rate upper bound decay parameter.
    pub fn new(
        m: &mut ParameterCollection,
        learning_rate_min: f32,
        learning_rate_max: f32,
        step_size: f32,
        gamma: f32,
    ) -> CyclicalSGDTrainer {
        unsafe {
            let mut trainer_ptr: *mut dynet_sys::dynetTrainer_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateCyclicalSGDTrainer(
                m.as_mut_ptr(),
                learning_rate_min,
                learning_rate_max,
                step_size,
                gamma,
                &mut trainer_ptr
            ));
            CyclicalSGDTrainer::from_raw(trainer_ptr, true)
        }
    }

    /// Creates a new `CyclicalSGDTrainer` with default settings.
    ///
    /// This is equivalent to `CyclicalSGDTrainer::new(m, 0.01, 0.1, 2000.0, 1.0)`.
    pub fn default(m: &mut ParameterCollection) -> CyclicalSGDTrainer {
        CyclicalSGDTrainer::new(m, 0.01, 0.1, 2000.0, 1.0)
    }
}

/// Stochastic gradient descent with momentum.
#[derive(Debug)]
pub struct MomentumSGDTrainer {
    inner: NonNull<dynet_sys::dynetTrainer_t>,
}

impl_trainer!(MomentumSGDTrainer);

impl MomentumSGDTrainer {
    /// Creates a new `MomentumSGDTrainer`.
    ///
    /// # Arguments
    ///
    /// * m - ParameterCollection to be trained.
    /// * learning_rate - Initial learning rate.
    /// * mom - Momentum.
    pub fn new(m: &mut ParameterCollection, learning_rate: f32, mom: f32) -> MomentumSGDTrainer {
        unsafe {
            let mut trainer_ptr: *mut dynet_sys::dynetTrainer_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateMomentumSGDTrainer(
                m.as_mut_ptr(),
                learning_rate,
                mom,
                &mut trainer_ptr
            ));
            MomentumSGDTrainer::from_raw(trainer_ptr, true)
        }
    }

    /// Creates a new `MomentumSGDTrainer` with default settings.
    ///
    /// This is equivalent to `MomentumSGDTrainer::new(m, 0.01, 0.9)`.
    pub fn default(m: &mut ParameterCollection) -> MomentumSGDTrainer {
        MomentumSGDTrainer::new(m, 0.01, 0.9)
    }
}

/// Adagrad optimizer.
#[derive(Debug)]
pub struct AdagradTrainer {
    inner: NonNull<dynet_sys::dynetTrainer_t>,
}

impl_trainer!(AdagradTrainer);

impl AdagradTrainer {
    /// Creates a new `AdagradTrainer`.
    ///
    /// # Arguments
    ///
    /// * m - ParameterCollection to be trained.
    /// * learning_rate - Initial learning rate.
    /// * eps - Bias parameter __epsilon__ in the adagrad formula.
    pub fn new(m: &mut ParameterCollection, learning_rate: f32, eps: f32) -> AdagradTrainer {
        unsafe {
            let mut trainer_ptr: *mut dynet_sys::dynetTrainer_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateAdagradTrainer(
                m.as_mut_ptr(),
                learning_rate,
                eps,
                &mut trainer_ptr
            ));
            AdagradTrainer::from_raw(trainer_ptr, true)
        }
    }

    /// Creates a new `AdagradTrainer` with default settings.
    ///
    /// This is equivalent to `AdagradTrainer::new(m, 0.1, 1e-20)`.
    pub fn default(m: &mut ParameterCollection) -> AdagradTrainer {
        AdagradTrainer::new(m, 0.1, 1e-20)
    }
}

/// AdaDelta optimizer.
#[derive(Debug)]
pub struct AdadeltaTrainer {
    inner: NonNull<dynet_sys::dynetTrainer_t>,
}

impl_trainer!(AdadeltaTrainer);

impl AdadeltaTrainer {
    /// Creates a new `AdadeltaTrainer`.
    ///
    /// # Arguments
    ///
    /// * m - ParameterCollection to be trained.
    /// * eps - Bias parameter __epsilon__ in the adadelta formula.
    /// * rho - Update parameter for the moving average of updates in the numerator.
    pub fn new(m: &mut ParameterCollection, eps: f32, rho: f32) -> AdadeltaTrainer {
        unsafe {
            let mut trainer_ptr: *mut dynet_sys::dynetTrainer_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateAdadeltaTrainer(
                m.as_mut_ptr(),
                eps,
                rho,
                &mut trainer_ptr
            ));
            AdadeltaTrainer::from_raw(trainer_ptr, true)
        }
    }

    /// Creates a new `AdadeltaTrainer` with default settings.
    ///
    /// This is equivalent to `AdadeltaTrainer::new(m, 1e-6. 0.95)`.
    pub fn default(m: &mut ParameterCollection) -> AdadeltaTrainer {
        AdadeltaTrainer::new(m, 1e-6, 0.95)
    }
}

/// RMSProp optimizer.
#[derive(Debug)]
pub struct RMSPropTrainer {
    inner: NonNull<dynet_sys::dynetTrainer_t>,
}

impl_trainer!(RMSPropTrainer);

impl RMSPropTrainer {
    /// Creates a new `RMSPropTrainer`.
    ///
    /// # Arguments
    ///
    /// * m - ParameterCollection to be trained.
    /// * learning_rate - Initial learning rate.
    /// * eps - Bias parameter __epsilon__ in the adagrad formula.
    /// * rho - Update parameter for the moving average (`rho = 0` is equivalent to using Adagrad).
    pub fn new(
        m: &mut ParameterCollection,
        learning_rate: f32,
        eps: f32,
        rho: f32,
    ) -> RMSPropTrainer {
        unsafe {
            let mut trainer_ptr: *mut dynet_sys::dynetTrainer_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateRMSPropTrainer(
                m.as_mut_ptr(),
                learning_rate,
                eps,
                rho,
                &mut trainer_ptr
            ));
            RMSPropTrainer::from_raw(trainer_ptr, true)
        }
    }

    /// Creates a new `RMSPropTrainer` with default settings.
    ///
    /// This is equivalent to `RMSPropTrainer::new(m, 0.1, 1e-20. 0.95)`.
    pub fn default(m: &mut ParameterCollection) -> RMSPropTrainer {
        RMSPropTrainer::new(m, 0.1, 1e-20, 0.95)
    }
}

/// Adam optimizer.
#[derive(Debug)]
pub struct AdamTrainer {
    inner: NonNull<dynet_sys::dynetTrainer_t>,
}

impl_trainer!(AdamTrainer);

impl AdamTrainer {
    /// Creates a new `AdamTrainer`.
    ///
    /// # Arguments
    ///
    /// * m - ParameterCollection to be trained.
    /// * learning_rate - Initial learning rate.
    /// * beta_1 - Moving average parameter for the mean.
    /// * beta_2 - Moving average parameter for the variance.
    /// * eps - Bias parameter __epsilon__.
    pub fn new(
        m: &mut ParameterCollection,
        learning_rate: f32,
        beta_1: f32,
        beta_2: f32,
        eps: f32,
    ) -> AdamTrainer {
        unsafe {
            let mut trainer_ptr: *mut dynet_sys::dynetTrainer_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateAdamTrainer(
                m.as_mut_ptr(),
                learning_rate,
                beta_1,
                beta_2,
                eps,
                &mut trainer_ptr
            ));
            AdamTrainer::from_raw(trainer_ptr, true)
        }
    }

    /// Creates a new `AdamTrainer` with default settings.
    ///
    /// This is equivalent to `AdamTrainer::new(m, 0.001, 0.9, 0.999, 1e-8)`.
    pub fn default(m: &mut ParameterCollection) -> AdamTrainer {
        AdamTrainer::new(m, 0.001, 0.9, 0.999, 1e-8)
    }
}

/// Amsgrad optimizer.
#[derive(Debug)]
pub struct AmsgradTrainer {
    inner: NonNull<dynet_sys::dynetTrainer_t>,
}

impl_trainer!(AmsgradTrainer);

impl AmsgradTrainer {
    /// Creates a new `AmsgradTrainer`.
    ///
    /// # Arguments
    ///
    /// * m - ParameterCollection to be trained.
    /// * learning_rate - Initial learning rate.
    /// * beta_1 - Moving average parameter for the mean.
    /// * beta_2 - Moving average parameter for the variance.
    /// * eps - Bias parameter __epsilon__.
    pub fn new(
        m: &mut ParameterCollection,
        learning_rate: f32,
        beta_1: f32,
        beta_2: f32,
        eps: f32,
    ) -> AmsgradTrainer {
        unsafe {
            let mut trainer_ptr: *mut dynet_sys::dynetTrainer_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateAmsgradTrainer(
                m.as_mut_ptr(),
                learning_rate,
                beta_1,
                beta_2,
                eps,
                &mut trainer_ptr
            ));
            AmsgradTrainer::from_raw(trainer_ptr, true)
        }
    }

    /// Creates a new `AmsgradTrainer` with default settings.
    ///
    /// This is equivalent to `AmsgradTrainer::new(m, 0.001, 0.9, 0.999, 1e-8)`.
    pub fn default(m: &mut ParameterCollection) -> AmsgradTrainer {
        AmsgradTrainer::new(m, 0.001, 0.9, 0.999, 1e-8)
    }
}
