use std::ffi::CString;
use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, Result, Wrap};

/// `ParameterInit` trait
pub trait ParameterInit: Wrap<dynet_sys::dynetParameterInit_t> {}

macro_rules! impl_initializer {
    ($name:ident) => {
        impl_wrap_owned!($name, dynetParameterInit_t);
        impl_drop!($name, dynetDeleteParameterInit);
        impl ParameterInit for $name {}
    };
}

/// An implementation of `ParameterInit` trait that initializes parameters with samples from a
/// normal distribution.
#[derive(Debug)]
pub struct ParameterInitNormal {
    inner: NonNull<dynet_sys::dynetParameterInit_t>,
}

impl_initializer!(ParameterInitNormal);

impl ParameterInitNormal {
    /// Creates a new `ParameterInitNormal`.
    ///
    /// # Arguments
    ///
    /// * m - Mean of the gaussian distribution.
    /// * v - Variance of the gaussian distribution.
    pub fn new(m: f32, v: f32) -> ParameterInitNormal {
        unsafe {
            let mut pinit_ptr: *mut dynet_sys::dynetParameterInit_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateParameterInitNormal(
                m,
                v,
                &mut pinit_ptr,
            ));
            ParameterInitNormal::from_raw(pinit_ptr, true)
        }
    }
}

impl Default for ParameterInitNormal {
    fn default() -> ParameterInitNormal {
        ParameterInitNormal::new(0.0, 1.0)
    }
}

/// An implementation of `ParameterInit` trait that initializes parameters with samples from a
/// uniform distribution.
#[derive(Debug)]
pub struct ParameterInitUniform {
    inner: NonNull<dynet_sys::dynetParameterInit_t>,
}

impl_initializer!(ParameterInitUniform);

impl ParameterInitUniform {
    /// Creates a new `ParameterInitUniform`.
    ///
    /// # Arguments
    ///
    /// * l - Lower bound of the interval.
    /// * r - Upper bound of the interval.
    ///
    /// # Panics
    ///
    /// Panics if `l` and `r` are the same value.
    pub fn new(l: f32, r: f32) -> ParameterInitUniform {
        unsafe {
            let mut pinit_ptr: *mut dynet_sys::dynetParameterInit_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateParameterInitUniform(
                l,
                r,
                &mut pinit_ptr,
            ));
            ParameterInitUniform::from_raw(pinit_ptr, true)
        }
    }
}

/// An implementation of `ParameterInit` trait that initializes parameters with a constant value.
#[derive(Debug)]
pub struct ParameterInitConst {
    inner: NonNull<dynet_sys::dynetParameterInit_t>,
}

impl_initializer!(ParameterInitConst);

impl ParameterInitConst {
    /// Creates a new `ParameterInitConst`.
    pub fn new(c: f32) -> ParameterInitConst {
        unsafe {
            let mut pinit_ptr: *mut dynet_sys::dynetParameterInit_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateParameterInitConst(c, &mut pinit_ptr));
            ParameterInitConst::from_raw(pinit_ptr, true)
        }
    }
}

/// An implementation of `ParameterInit` trait that initializes parameters as the identity.
///
/// This will cause panics if used on non square matrices.
#[derive(Debug)]
pub struct ParameterInitIdentity {
    inner: NonNull<dynet_sys::dynetParameterInit_t>,
}

impl_initializer!(ParameterInitIdentity);

impl ParameterInitIdentity {
    /// Creates a new `ParameterInitIdentity`.
    pub fn new() -> ParameterInitIdentity {
        unsafe {
            let mut pinit_ptr: *mut dynet_sys::dynetParameterInit_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateParameterInitIdentity(&mut pinit_ptr));
            ParameterInitIdentity::from_raw(pinit_ptr, true)
        }
    }
}

impl Default for ParameterInitIdentity {
    fn default() -> ParameterInitIdentity {
        ParameterInitIdentity::new()
    }
}

/// An implementation of `ParameterInit` trait that initializes parameters by the Glorot
/// (uniform) initialization.
///
/// This initializes parameters with the methods described in [Glorot, 2010](http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf?hc_location=ufi).
#[derive(Debug)]
pub struct ParameterInitGlorot {
    inner: NonNull<dynet_sys::dynetParameterInit_t>,
}

impl_initializer!(ParameterInitGlorot);

impl ParameterInitGlorot {
    /// Creates a new `ParameterInitGlorot`.
    ///
    /// # Arguments
    ///
    /// * is_lookup - Boolean value identifying the parameter as a LookupParameter.
    /// * gain - Scaling parameter.
    pub fn new(is_lookup: bool, gain: f32) -> ParameterInitGlorot {
        unsafe {
            let mut pinit_ptr: *mut dynet_sys::dynetParameterInit_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateParameterInitGlorot(
                is_lookup as u32,
                gain,
                &mut pinit_ptr,
            ));
            ParameterInitGlorot::from_raw(pinit_ptr, true)
        }
    }
}

impl Default for ParameterInitGlorot {
    fn default() -> ParameterInitGlorot {
        ParameterInitGlorot::new(false, 1.0)
    }
}

/// An implementation of `ParameterInit` trait that initializes parameters by the Saxe
/// initialization.
///
/// This initializes parameters with the methods described in [Saxe et al., 2014](https://arxiv.org/abs/1312.6120).
#[derive(Debug)]
pub struct ParameterInitSaxe {
    inner: NonNull<dynet_sys::dynetParameterInit_t>,
}

impl_initializer!(ParameterInitSaxe);

impl ParameterInitSaxe {
    /// Creates a new `ParameterInitSaxe`.
    ///
    /// # Arguments
    ///
    /// * gain - Scaling parameter.
    pub fn new(gain: f32) -> ParameterInitSaxe {
        unsafe {
            let mut pinit_ptr: *mut dynet_sys::dynetParameterInit_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateParameterInitSaxe(
                gain,
                &mut pinit_ptr,
            ));
            ParameterInitSaxe::from_raw(pinit_ptr, true)
        }
    }
}

impl Default for ParameterInitSaxe {
    fn default() -> ParameterInitSaxe {
        ParameterInitSaxe::new(1.0)
    }
}

/// An implementation of `ParameterInit` trait that initializes parameters from a file.
#[derive(Debug)]
pub struct ParameterInitFromFile {
    inner: NonNull<dynet_sys::dynetParameterInit_t>,
}

impl_initializer!(ParameterInitFromFile);

impl ParameterInitFromFile {
    /// Creates a new `ParameterInitFromFile`.
    pub fn new(f: &str) -> ParameterInitFromFile {
        unsafe {
            let f_c = CString::new(f).unwrap();
            let mut pinit_ptr: *mut dynet_sys::dynetParameterInit_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateParameterInitFromFile(
                f_c.as_ptr(),
                &mut pinit_ptr,
            ));
            ParameterInitFromFile::from_raw(pinit_ptr, true)
        }
    }
}

/// An implementation of `ParameterInit` trait that initializes parameters from a slice.
#[derive(Debug)]
pub struct ParameterInitFromSlice {
    inner: NonNull<dynet_sys::dynetParameterInit_t>,
}

impl_initializer!(ParameterInitFromSlice);

impl ParameterInitFromSlice {
    /// Creates a new `ParameterInitFromSlice`.
    pub fn new(v: &[f32]) -> ParameterInitFromSlice {
        unsafe {
            let mut pinit_ptr: *mut dynet_sys::dynetParameterInit_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateParameterInitFromVector(
                v.as_ptr(),
                v.len(),
                &mut pinit_ptr,
            ));
            ParameterInitFromSlice::from_raw(pinit_ptr, true)
        }
    }
}
