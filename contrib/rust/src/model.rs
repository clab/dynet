use std::ffi::CString;
use std::io as std_io;
use std::path::Path;
use std::ptr::{self, NonNull};

use dynet_sys;

use super::{
    ApiResult, Device, Dim, Load, ParameterInit, Result, Save, Tensor, TextFileLoader,
    TextFileSaver, Wrap,
};

/// A struct to represent a trainable parameter.
#[derive(Debug)]
pub struct Parameter {
    inner: NonNull<dynet_sys::dynetParameter_t>,
    owned: bool,
}

impl_wrap!(Parameter, dynetParameter_t);
impl_drop!(Parameter, dynetDeleteParameter);

impl Parameter {
    /// Fills values with zero.
    pub fn zero(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetFillParameterWithZeros(self.as_mut_ptr()));
        }
    }

    /// Returns the dim of the parameter.
    pub fn dim(&self) -> Dim {
        unsafe {
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetParameterDim(self.as_ptr(), &mut dim_ptr));
            Dim::from_raw(dim_ptr, true)
        }
    }

    /// Returns values of the parameter as a tensor.
    pub fn values(&mut self) -> Tensor {
        unsafe {
            let mut tensor_ptr: *mut dynet_sys::dynetTensor_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetParameterValues(
                self.as_mut_ptr(),
                &mut tensor_ptr,
            ));
            Tensor::from_raw(tensor_ptr, false)
        }
    }

    /// Returns gradients of the parameter as a tensor.
    pub fn gradients(&mut self) -> Tensor {
        unsafe {
            let mut tensor_ptr: *mut dynet_sys::dynetTensor_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetParameterGradients(
                self.as_mut_ptr(),
                &mut tensor_ptr,
            ));
            Tensor::from_raw(tensor_ptr, false)
        }
    }

    /// Sets update status of the parameter.
    ///
    /// If `b` is true, this parameter will be updated during training.
    pub fn set_updated(&mut self, b: bool) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetParameterUpdated(
                self.as_mut_ptr(),
                b as u32,
            ));
        }
    }

    /// Returns update status of the parameter.
    pub fn is_updated(&self) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetIsParameterUpdated(
                self.as_ptr(),
                &mut retval,
            ));
            retval == 1
        }
    }
}

impl Save for Parameter {
    fn save<P: AsRef<Path>>(&self, path: P) -> std_io::Result<()> {
        TextFileSaver::new(path, false).and_then(|mut saver| saver.save_parameter(self, None))
    }
}

impl Load for Parameter {
    fn load<P: AsRef<Path>>(&mut self, path: P) -> std_io::Result<()> {
        TextFileLoader::new(path).and_then(|mut loader| loader.populate_parameter(self, None))
    }
}

/// A struct to represent a trainable lookup parameter.
#[derive(Debug)]
pub struct LookupParameter {
    inner: NonNull<dynet_sys::dynetLookupParameter_t>,
    owned: bool,
}

impl_wrap!(LookupParameter, dynetLookupParameter_t);
impl_drop!(LookupParameter, dynetDeleteLookupParameter);

impl LookupParameter {
    /// Initializes one paticular column of the values in the parameter.
    pub fn initialize(&mut self, index: u32, val: &[f32]) {
        unsafe {
            check_api_status!(dynet_sys::dynetInitializeLookupParameter(
                self.as_mut_ptr(),
                index,
                val.as_ptr(),
                val.len(),
            ));
        }
    }

    /// Fills values with zero.
    pub fn zero(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetFillLookupParameterWithZeros(
                self.as_mut_ptr()
            ));
        }
    }

    /// Returns the dim of the parameter.
    pub fn dim(&self) -> Dim {
        unsafe {
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetLookupParameterDim(
                self.as_ptr(),
                &mut dim_ptr,
            ));
            Dim::from_raw(dim_ptr, true)
        }
    }

    /// Returns values of the parameter as a vector of tensors.
    pub fn values(&mut self) -> Vec<Tensor> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(dynet_sys::dynetGetLookupParameterValues(
                self.as_mut_ptr(),
                ptr::null_mut(),
                &mut size,
            ));
            let mut tensor_ptrs = vec![ptr::null_mut(); size];
            check_api_status!(dynet_sys::dynetGetLookupParameterValues(
                self.as_mut_ptr(),
                tensor_ptrs.as_mut_ptr(),
                &mut size,
            ));
            tensor_ptrs
                .into_iter()
                .map(|tensor_ptr| Tensor::from_raw(tensor_ptr, false))
                .collect()
        }
    }

    /// Sets update status of the parameter.
    ///
    /// If `b` is true, this parameter will be updated during training.
    pub fn set_updated(&mut self, b: bool) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetLookupParameterUpdated(
                self.as_mut_ptr(),
                b as u32,
            ));
        }
    }

    /// Returns update status of the parameter.
    pub fn is_updated(&self) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetIsLookupParameterUpdated(
                self.as_ptr(),
                &mut retval,
            ));
            retval == 1
        }
    }
}

impl Save for LookupParameter {
    fn save<P: AsRef<Path>>(&self, path: P) -> std_io::Result<()> {
        TextFileSaver::new(path, false)
            .and_then(|mut saver| saver.save_lookup_parameter(self, None))
    }
}

impl Load for LookupParameter {
    fn load<P: AsRef<Path>>(&mut self, path: P) -> std_io::Result<()> {
        TextFileLoader::new(path)
            .and_then(|mut loader| loader.populate_lookup_parameter(self, None))
    }
}

/// A struct to represent a collection of parameters.
///
/// # Examples
///
/// ```
/// # use dynet::{DynetParams, ParameterCollection, ParameterInit, ParameterInitGlorot};
/// dynet::initialize(&mut DynetParams::default());
///
/// let mut model = ParameterCollection::new();
///
/// let initializer = ParameterInitGlorot::default();
/// let mut p_W = model.add_parameters([8, 2], &initializer);
/// let mut p_b = model.add_parameters([8], &initializer);
/// ```
#[derive(Debug)]
pub struct ParameterCollection {
    inner: NonNull<dynet_sys::dynetParameterCollection_t>,
    owned: bool,
}

impl_wrap!(ParameterCollection, dynetParameterCollection_t);
impl_drop!(ParameterCollection, dynetDeleteParameterCollection);

impl ParameterCollection {
    /// Creates a new `ParameterCollection`.
    pub fn new() -> ParameterCollection {
        unsafe {
            let mut pc_ptr: *mut dynet_sys::dynetParameterCollection_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateParameterCollection(&mut pc_ptr));
            ParameterCollection::from_raw(pc_ptr, true)
        }
    }

    /// Returns the L2 norm of the gradient in the ParameterCollection.
    pub fn gradient_l2_norm(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(dynet_sys::dynetGetParameterCollectionGradientL2Norm(
                self.as_ptr(),
                &mut retval,
            ));
            retval
        }
    }

    /// Sets weight decay coefficient for parameters.
    pub fn set_weight_decay(&mut self, lambda: f32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetParameterCollectionWeightDecayLambda(
                self.as_mut_ptr(),
                lambda
            ));
        }
    }

    /// Gets weight decay lambda value.
    pub fn get_weight_decay(&self) -> f32 {
        unsafe {
            let mut retval: f32 = 0.0;
            check_api_status!(dynet_sys::dynetGetParameterCollectionWeightDecayLambda(
                self.as_ptr(),
                &mut retval,
            ));
            retval
        }
    }

    /// Adds parameters.
    pub fn add_parameters<D: Into<Dim>, I: ParameterInit>(&mut self, d: D, init: &I) -> Parameter {
        self.add_named_parameters_on(d, init, None, None)
    }

    /// Adds parameters with a name on an explicit device.
    ///
    /// # Panics
    ///
    /// Panics if `name` has an invalid character including `/` and `_`.
    pub fn add_named_parameters_on<D: Into<Dim>, I: ParameterInit>(
        &mut self,
        d: D,
        init: &I,
        name: Option<&str>,
        device: Option<&mut Device>,
    ) -> Parameter {
        unsafe {
            let mut param_ptr: *mut dynet_sys::dynetParameter_t = ptr::null_mut();
            let name_c = CString::new(name.unwrap_or("")).unwrap();
            check_api_status!(dynet_sys::dynetAddParametersToParameterCollection(
                self.as_mut_ptr(),
                d.into().as_ptr(),
                init.as_ptr(),
                name_c.as_ptr(),
                device.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
                &mut param_ptr,
            ));
            Parameter::from_raw(param_ptr, true)
        }
    }

    /// Adds lookup parameters.
    pub fn add_lookup_parameters<D: Into<Dim>, I: ParameterInit>(
        &mut self,
        n: u32,
        d: D,
        init: &I,
    ) -> LookupParameter {
        self.add_named_lookup_parameters_on(n, d, init, None, None)
    }

    /// Adds lookup parameters with a name on an explicit device.
    ///
    /// # Arguments
    ///
    /// * n - Number of lookup indices.
    /// * d - Dimension of each embedding.
    /// * init - Initializer.
    /// * name - Name of the parameter.
    /// * device - Device placement for the parameter.
    ///
    /// # Panics
    ///
    /// Panics if `name` has an invalid character including `/` and `_`.
    pub fn add_named_lookup_parameters_on<D: Into<Dim>, I: ParameterInit>(
        &mut self,
        n: u32,
        d: D,
        init: &I,
        name: Option<&str>,
        device: Option<&mut Device>,
    ) -> LookupParameter {
        unsafe {
            let mut param_ptr: *mut dynet_sys::dynetLookupParameter_t = ptr::null_mut();
            let name_c = CString::new(name.unwrap_or("")).unwrap();
            check_api_status!(dynet_sys::dynetAddLookupParametersToParameterCollection(
                self.as_mut_ptr(),
                n,
                d.into().as_ptr(),
                init.as_ptr(),
                name_c.as_ptr(),
                device.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut()),
                &mut param_ptr,
            ));
            LookupParameter::from_raw(param_ptr, true)
        }
    }

    /// Adds a subcollection.
    pub fn add_subcollection(&mut self) -> ParameterCollection {
        self.add_named_subcollection(None)
    }

    /// Adds a subcollection with a name.
    ///
    /// # Panics
    ///
    /// Panics if `name` has an invalid character including `/` and `_`.
    pub fn add_named_subcollection(&mut self, name: Option<&str>) -> ParameterCollection {
        unsafe {
            let mut pc_ptr: *mut dynet_sys::dynetParameterCollection_t = ptr::null_mut();
            let name_c = CString::new(name.unwrap_or("")).unwrap();
            check_api_status!(dynet_sys::dynetAddSubcollectionToParameterCollection(
                self.as_mut_ptr(),
                name_c.as_ptr(),
                &mut pc_ptr,
            ));
            ParameterCollection::from_raw(pc_ptr, true)
        }
    }

    /// Returns the total number of tunable parameters.
    pub fn parameter_count(&self) -> usize {
        unsafe {
            let mut retval: usize = 0;
            check_api_status!(dynet_sys::dynetGetParameterCollectionParameterCount(
                self.as_ptr(),
                &mut retval,
            ));
            retval
        }
    }
}

impl Default for ParameterCollection {
    fn default() -> ParameterCollection {
        ParameterCollection::new()
    }
}

impl Save for ParameterCollection {
    fn save<P: AsRef<Path>>(&self, path: P) -> std_io::Result<()> {
        TextFileSaver::new(path, false).and_then(|mut saver| saver.save_model(self, None))
    }
}

impl Load for ParameterCollection {
    fn load<P: AsRef<Path>>(&mut self, path: P) -> std_io::Result<()> {
        TextFileLoader::new(path).and_then(|mut loader| loader.populate_model(self, None))
    }
}
