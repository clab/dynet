use std::env;
use std::ffi::CString;
use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, Result, Wrap};

/// A struct to represent general parameters for DyNet.
///
/// # Examples
///
/// ```
/// # use dynet::DynetParams;
/// dynet::initialize(
///     DynetParams::default()
///         .random_seed(0)
///         .mem_descriptor("256")
///         .weight_decay(0.001)
///         .autobatch(1)
/// );
/// ```
#[derive(Debug)]
pub struct DynetParams {
    inner: NonNull<dynet_sys::dynetDynetParams_t>,
}

impl_wrap_owned!(DynetParams, dynetDynetParams_t);
impl_drop!(DynetParams, dynetDeleteDynetParams);

impl DynetParams {
    /// Creates a new `DynetParams`.
    pub fn new() -> DynetParams {
        unsafe {
            let mut params_ptr: *mut dynet_sys::dynetDynetParams_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateDynetParams(&mut params_ptr));
            DynetParams::from_raw(params_ptr, true)
        }
    }

    /// Builds a `DynetParams` from command line arguments.
    pub fn from_args(enabled_shared_parameters: bool) -> DynetParams {
        let args: Vec<CString> = env::args().map(|s| CString::new(s).unwrap()).collect();
        unsafe {
            let mut arg_ptrs: Vec<_> = args.iter().map(|arg| arg.as_ptr() as *mut _).collect();
            let mut params_ptr: *mut dynet_sys::dynetDynetParams_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetExtractDynetParams(
                arg_ptrs.len() as i32,
                arg_ptrs.as_mut_ptr(),
                enabled_shared_parameters as u32,
                &mut params_ptr,
            ));
            DynetParams::from_raw(params_ptr, true)
        }
    }

    /// Sets the seed for random number generation.
    ///
    /// Default value: `0`
    pub fn random_seed(&mut self, seed: u32) -> &mut DynetParams {
        unsafe {
            check_api_status!(dynet_sys::dynetSetDynetParamsRandomSeed(
                self.as_mut_ptr(),
                seed,
            ));
        }
        self
    }

    /// Sets total memory to be allocated for DyNet.
    ///
    /// Default value: `"512"`
    pub fn mem_descriptor(&mut self, mem_descriptor: &str) -> &mut DynetParams {
        unsafe {
            let mem_descriptor_c = CString::new(mem_descriptor).unwrap();
            check_api_status!(dynet_sys::dynetSetDynetParamsMemDescriptor(
                self.as_mut_ptr(),
                mem_descriptor_c.as_ptr(),
            ));
        }
        self
    }

    /// Sets weight decay rate for L2 regularization.
    ///
    /// Default value: `0`
    pub fn weight_decay(&mut self, weight_decay: f32) -> &mut DynetParams {
        unsafe {
            check_api_status!(dynet_sys::dynetSetDynetParamsWeightDecay(
                self.as_mut_ptr(),
                weight_decay,
            ));
        }
        self
    }

    /// Specifies whether to autobatch or not.
    ///
    /// Default value: `0`
    pub fn autobatch(&mut self, autobatch: i32) -> &mut DynetParams {
        unsafe {
            check_api_status!(dynet_sys::dynetSetDynetParamsAutobatch(
                self.as_mut_ptr(),
                autobatch,
            ));
        }
        self
    }

    /// Specifies whether to show autobatch debug info or not.
    ///
    /// Default value: `0`
    pub fn profiling(&mut self, profiling: i32) -> &mut DynetParams {
        unsafe {
            check_api_status!(dynet_sys::dynetSetDynetParamsProfiling(
                self.as_mut_ptr(),
                profiling,
            ));
        }
        self
    }

    /// Specifies whether to share parameters or not.
    ///
    /// Default value: `false`
    pub fn shared_parameters(&mut self, enabled: bool) -> &mut DynetParams {
        unsafe {
            check_api_status!(dynet_sys::dynetSetDynetParamsSharedParameters(
                self.as_mut_ptr(),
                enabled as u32,
            ));
        }
        self
    }

    /// Specifies the number of requested GPUs.
    ///
    /// Default value: `-1`
    pub fn requested_gpus(&mut self, requested_gpus: i32) -> &mut DynetParams {
        unsafe {
            check_api_status!(dynet_sys::dynetSetDynetParamsRequestedGpus(
                self.as_mut_ptr(),
                requested_gpus,
            ));
        }
        self
    }
}

impl Default for DynetParams {
    fn default() -> DynetParams {
        DynetParams::new()
    }
}

/// Initializes DyNet.
///
/// # Panics
///
/// Panics if an invalid value is specified in `params`.
pub fn initialize(params: &mut DynetParams) {
    unsafe {
        check_api_status!(dynet_sys::dynetInitialize(params.as_mut_ptr()));
    }
}

/// Resets random number generators.
pub fn reset_rng(seed: u32) {
    unsafe {
        check_api_status!(dynet_sys::dynetResetRng(seed));
    }
}
