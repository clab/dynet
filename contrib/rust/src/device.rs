use std::ffi::CString;
use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, Result, Wrap};

/// A struct to represent a device.
///
/// # Examples
///
/// ```
/// # use dynet::{Device, DynetParams};
/// dynet::initialize(&mut DynetParams::default());
/// let device = Device::default();
/// ```
// TODO(chantera): write example with the function that takes `device` as an argument.
#[derive(Debug)]
pub struct Device {
    inner: NonNull<dynet_sys::dynetDevice_t>,
    owned: bool,
}

impl_wrap!(Device, dynetDevice_t);

impl Drop for Device {
    fn drop(&mut self) {}
}

impl Device {
    /// Retrieves a global device.
    ///
    /// Before calling this method, `dynet::initialize` must be executed to initialize devices.
    ///
    /// If `name` is empty, this returns the default device.
    ///
    /// # Panics
    ///
    /// Panics if the device `name` does not exist.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use dynet::{Device, DynetParams};
    /// // Run your program with the argument `--dynet-devices CPU,GPU:0,GPU:1`.
    /// dynet::initialize(&mut DynetParams::from_args(false));
    /// let dev0 = Device::global_device("GPU:0");
    /// let dev1 = Device::global_device("GPU:1");
    /// ```
    pub fn global_device(name: &str) -> Device {
        unsafe {
            let mut device_ptr: *mut dynet_sys::dynetDevice_t = ptr::null_mut();
            let name_c = CString::new(name).unwrap();
            check_api_status!(dynet_sys::dynetGetGlobalDevice(
                name_c.as_ptr(),
                &mut device_ptr,
            ));
            Device::from_raw(device_ptr, false)
        }
    }

    /// Returns the number of global devices.
    pub fn num_devices() -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetGetNumDevices(&mut retval));
            retval
        }
    }
}

impl Default for Device {
    fn default() -> Device {
        Device::global_device("")
    }
}
