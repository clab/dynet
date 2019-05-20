use std::ffi::CString;
use std::fmt;
use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, Dim, Result, Wrap};

/// A struct to represent a tensor.
///
/// # Examples
///
/// ```
/// # use dynet::{DynetParams, ParameterCollection, ParameterInitGlorot};
/// dynet::initialize(&mut DynetParams::from_args(false));
///
/// let mut m = ParameterCollection::new();
///
/// let initializer = ParameterInitGlorot::default();
/// let mut p_W = m.add_parameters([8, 2], &initializer);
/// let t_W = p_W.values();
/// println!("parameter W: dim={}, values=\n[\n{}\n]", t_W.dim(), t_W);
/// let v_W = t_W.as_vector();
/// ```
#[derive(Debug)]
pub struct Tensor {
    inner: NonNull<dynet_sys::dynetTensor_t>,
    owned: bool,
}

impl_wrap!(Tensor, dynetTensor_t);
impl_drop!(Tensor, dynetDeleteTensor);

impl Tensor {
    /// Returns the dim of the tensor.
    pub fn dim(&self) -> Dim {
        unsafe {
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetTensorDim(self.as_ptr(), &mut dim_ptr));
            Dim::from_raw(dim_ptr, true)
        }
    }

    /// Retrieves one internal value in the tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor has more than one element.
    pub fn as_scalar(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(dynet_sys::dynetEvaluateTensorAsScalar(
                self.as_ptr(),
                &mut retval,
            ));
            retval
        }
    }

    /// Retrieves internal values in the tensor as a vector.
    ///
    /// For higher order tensors this returns the flattened value.
    pub fn as_vector(&self) -> Vec<f32> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(dynet_sys::dynetEvaluateTensorAsArray(
                self.as_ptr(),
                ptr::null_mut(),
                &mut size,
            ));
            let mut retval = vec![0f32; size];
            check_api_status!(dynet_sys::dynetEvaluateTensorAsArray(
                self.as_ptr(),
                retval.as_mut_ptr(),
                &mut size,
            ));
            retval
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(dynet_sys::dynetRepresentTensorAsString(
                self.as_ptr(),
                ptr::null_mut(),
                &mut size,
            ));
            let buffer = CString::new(vec![b'0'; size]).unwrap().into_raw();
            check_api_status!(dynet_sys::dynetRepresentTensorAsString(
                self.as_ptr(),
                buffer,
                &mut size,
            ));
            f.write_str(CString::from_raw(buffer).to_str().unwrap())
        }
    }
}
