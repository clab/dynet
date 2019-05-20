use std::ffi::CString;
use std::io as std_io;
use std::path::Path;
use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, LookupParameter, Parameter, ParameterCollection, Result, Wrap};

#[derive(Debug)]
pub struct TextFileSaver {
    inner: NonNull<dynet_sys::dynetTextFileSaver_t>,
}

impl_wrap_owned!(TextFileSaver, dynetTextFileSaver_t);
impl_drop!(TextFileSaver, dynetDeleteTextFileSaver);

impl TextFileSaver {
    pub fn new<P: AsRef<Path>>(path: P, append: bool) -> std_io::Result<TextFileSaver> {
        unsafe {
            let mut saver_ptr: *mut dynet_sys::dynetTextFileSaver_t = ptr::null_mut();
            let path_c = CString::new(path.as_ref().to_str().unwrap()).unwrap();
            let path_ptr = path_c.as_ptr();
            Result::from_api_status(
                dynet_sys::dynetCreateTextFileSaver(path_ptr, append as u32, &mut saver_ptr),
                (),
            ).map(|_| TextFileSaver::from_raw(saver_ptr, true))
                .map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }

    pub fn save_model(
        &mut self,
        model: &ParameterCollection,
        key: Option<&str>,
    ) -> std_io::Result<()> {
        unsafe {
            let key_c = CString::new(key.unwrap_or("")).unwrap();
            Result::from_api_status(
                dynet_sys::dynetSaveParameterCollection(
                    self.as_mut_ptr(),
                    model.as_ptr(),
                    key_c.as_ptr(),
                ),
                (),
            ).map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }

    pub fn save_parameter(&mut self, param: &Parameter, key: Option<&str>) -> std_io::Result<()> {
        unsafe {
            let key_c = CString::new(key.unwrap_or("")).unwrap();
            Result::from_api_status(
                dynet_sys::dynetSaveParameter(self.as_mut_ptr(), param.as_ptr(), key_c.as_ptr()),
                (),
            ).map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }

    pub fn save_lookup_parameter(
        &mut self,
        param: &LookupParameter,
        key: Option<&str>,
    ) -> std_io::Result<()> {
        unsafe {
            let key_c = CString::new(key.unwrap_or("")).unwrap();
            Result::from_api_status(
                dynet_sys::dynetSaveLookupParameter(
                    self.as_mut_ptr(),
                    param.as_ptr(),
                    key_c.as_ptr(),
                ),
                (),
            ).map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }
}

pub trait Save {
    fn save<P: AsRef<Path>>(&self, path: P) -> std_io::Result<()>;
}

#[derive(Debug)]
pub struct TextFileLoader {
    inner: NonNull<dynet_sys::dynetTextFileLoader>,
}

impl_wrap_owned!(TextFileLoader, dynetTextFileLoader_t);
impl_drop!(TextFileLoader, dynetDeleteTextFileLoader);

impl TextFileLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> std_io::Result<TextFileLoader> {
        unsafe {
            let mut loader_ptr: *mut dynet_sys::dynetTextFileLoader_t = ptr::null_mut();
            let path_c = CString::new(path.as_ref().to_str().unwrap()).unwrap();
            let path_ptr = path_c.as_ptr();
            Result::from_api_status(
                dynet_sys::dynetCreateTextFileLoader(path_ptr, &mut loader_ptr),
                (),
            ).map(|_| TextFileLoader::from_raw(loader_ptr, true))
                .map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }

    pub fn populate_model(
        &mut self,
        model: &mut ParameterCollection,
        key: Option<&str>,
    ) -> std_io::Result<()> {
        unsafe {
            let key_c = CString::new(key.unwrap_or("")).unwrap();
            Result::from_api_status(
                dynet_sys::dynetPopulateParameterCollection(
                    self.as_mut_ptr(),
                    model.as_mut_ptr(),
                    key_c.as_ptr(),
                ),
                (),
            ).map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }

    pub fn populate_parameter(
        &mut self,
        param: &mut Parameter,
        key: Option<&str>,
    ) -> std_io::Result<()> {
        unsafe {
            let key_c = CString::new(key.unwrap_or("")).unwrap();
            Result::from_api_status(
                dynet_sys::dynetPopulateParameter(
                    self.as_mut_ptr(),
                    param.as_mut_ptr(),
                    key_c.as_ptr(),
                ),
                (),
            ).map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }

    pub fn populate_lookup_parameter(
        &mut self,
        param: &mut LookupParameter,
        key: Option<&str>,
    ) -> std_io::Result<()> {
        unsafe {
            let key_c = CString::new(key.unwrap_or("")).unwrap();
            Result::from_api_status(
                dynet_sys::dynetPopulateLookupParameter(
                    self.as_mut_ptr(),
                    param.as_mut_ptr(),
                    key_c.as_ptr(),
                ),
                (),
            ).map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }

    pub fn load_param(
        &mut self,
        model: &mut ParameterCollection,
        key: &str,
    ) -> std_io::Result<Parameter> {
        unsafe {
            let mut param_ptr: *mut dynet_sys::dynetParameter_t = ptr::null_mut();
            let key_c = CString::new(key).unwrap();
            Result::from_api_status(
                dynet_sys::dynetLoadParameterFromParameterCollection(
                    self.as_mut_ptr(),
                    model.as_mut_ptr(),
                    key_c.as_ptr(),
                    &mut param_ptr,
                ),
                (),
            ).map(|_| Parameter::from_raw(param_ptr, true))
                .map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }

    pub fn load_lookup_param(
        &mut self,
        model: &mut ParameterCollection,
        key: &str,
    ) -> std_io::Result<LookupParameter> {
        unsafe {
            let mut param_ptr: *mut dynet_sys::dynetLookupParameter_t = ptr::null_mut();
            let key_c = CString::new(key).unwrap();
            Result::from_api_status(
                dynet_sys::dynetLoadLookupParameterFromParameterCollection(
                    self.as_mut_ptr(),
                    model.as_mut_ptr(),
                    key_c.as_ptr(),
                    &mut param_ptr,
                ),
                (),
            ).map(|_| LookupParameter::from_raw(param_ptr, true))
                .map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }
}

pub trait Load {
    fn load<P: AsRef<Path>>(&mut self, path: P) -> std_io::Result<()>;
}
