use std::cmp::{Eq, PartialEq};
use std::ffi::CString;
use std::fmt;
use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, Result, Wrap};

/// A struct to store information about the dimensionality of expressions.
///
/// Batch dimension is treated separately from standard dimension.
#[derive(Debug)]
pub struct Dim {
    inner: NonNull<dynet_sys::dynetDim_t>,
}

impl_wrap_owned!(Dim, dynetDim_t);
impl_drop!(Dim, dynetDeleteDim);

impl Dim {
    /// Creates a new `Dim`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let dim = Dim::new();
    ///
    /// // The dim contains neither dimensions nor batches.
    /// assert_eq!(dim.ndims(), 0);
    /// assert_eq!(dim.batch_elems(), 1);
    /// ```
    pub fn new() -> Dim {
        unsafe {
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateDim(&mut dim_ptr));
            Dim::from_raw(dim_ptr, true)
        }
    }

    /// Creates a new `Dim` with the specified dimensions and batch size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let dim = Dim::with_dimensions(&[2, 4, 6], 8);
    /// assert_eq!(dim.ndims(), 3);
    /// assert_eq!(dim.batch_elems(), 8);
    /// ```
    pub fn with_dimensions(x: &[u32], b: u32) -> Dim {
        unsafe {
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateDimWithDimensionsAndBatch(
                x.as_ptr(),
                x.len(),
                b,
                &mut dim_ptr,
            ));
            Dim::from_raw(dim_ptr, true)
        }
    }

    /// Returns the total size of the dim.
    ///
    /// The returned value is equal to the product of the number of elements in a batch and the
    /// number of batches (`batch_size` * `batch_elems`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let dim = Dim::from(([2, 3], 4));
    /// assert_eq!(dim.size(), 24); // (2 * 3) * 4
    /// assert_eq!(dim.size(), dim.batch_size() * dim.batch_elems());
    /// ```
    pub fn size(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetGetDimTotalSize(self.as_ptr(), &mut retval));
            retval
        }
    }

    /// Returns the size of a batch.
    ///
    /// The returned value is equal to the product of all dimensions with in a batch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let dim = Dim::from([2, 3, 4]);
    /// assert_eq!(dim.batch_size(), 24); // 2 * 3 * 4
    /// ```
    pub fn batch_size(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetGetDimBatchSize(self.as_ptr(), &mut retval));
            retval
        }
    }

    /// Returns sum of all dimensions with in a batch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let dim = Dim::from([2, 3, 4]);
    /// assert_eq!(dim.sum_dims(), 9); // 2 + 3 + 4
    /// ```
    pub fn sum_dims(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetSumDimDimensions(self.as_ptr(), &mut retval));
            retval
        }
    }

    /// Truncates trailing dimensions of one.
    ///
    /// This iterates all the dimensions of `Dim` and stops at last dimension of one.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let dim = Dim::from([2, 3, 1, 1]);
    /// assert_eq!(dim.ndims(), 4); // [2, 3, 1, 1]
    /// let dim_truncated = dim.truncate();
    /// assert_eq!(dim_truncated.ndims(), 2); // [2, 3]
    /// ```
    pub fn truncate(&self) -> Dim {
        unsafe {
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetTruncateDim(self.as_ptr(), &mut dim_ptr));
            Dim::from_raw(dim_ptr, true)
        }
    }

    /// Changes the number of dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let mut dim = Dim::from([2, 3, 4, 5]);
    /// assert_eq!(dim.ndims(), 4); // [2, 3, 4, 5]
    /// dim.resize(2);
    /// assert_eq!(dim.ndims(), 2); // [2, 3]
    /// ```
    pub fn resize(&mut self, i: u32) {
        unsafe {
            check_api_status!(dynet_sys::dynetResizeDim(self.as_mut_ptr(), i));
        }
    }

    /// Returns the number of dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let dim = Dim::from([2, 3, 1, 1, 1, 5]);
    /// assert_eq!(dim.ndims(), 6);
    /// ```
    pub fn ndims(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetGetDimNDimensions(
                self.as_ptr(),
                &mut retval,
            ));
            retval
        }
    }

    /// Returns the size of the first dimension.
    ///
    /// This returns `0` if the dim has no dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let d1 = Dim::from([4, 5, 6]);
    /// assert_eq!(d1.rows(), 4);
    ///
    /// let d2 = Dim::new();
    /// assert_eq!(d2.rows(), 0);
    /// ```
    pub fn rows(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetGetDimRows(self.as_ptr(), &mut retval));
            retval
        }
    }

    /// Returns the size of the second dimension.
    ///
    /// This returns `1` if the dim has only one dimension or no dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let d1 = Dim::from([4, 5, 6]);
    /// assert_eq!(d1.cols(), 5);
    ///
    /// let d2 = Dim::from([4]);
    /// assert_eq!(d2.cols(), 1);
    ///
    /// let d3 = Dim::from([]);
    /// assert_eq!(d3.cols(), 1);
    /// ```
    pub fn cols(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetGetDimCols(self.as_ptr(), &mut retval));
            retval
        }
    }

    /// Returns the batch dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let dim = Dim::from(([4, 5, 6], 8));
    /// assert_eq!(dim.batch_elems(), 8);
    /// ```
    pub fn batch_elems(&self) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetGetDimBatchElems(self.as_ptr(), &mut retval));
            retval
        }
    }

    /// Returns the specific dimension.
    ///
    /// This returns `1` if the specifie dimension does not exist.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let dim = Dim::from(([3, 4, 5], 8));
    /// assert_eq!(dim.get(2), 5);
    /// assert_eq!(dim.get(10), 1);
    /// ```
    pub fn get(&self, i: u32) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetGetDimDimensionSize(
                self.as_ptr(),
                i,
                &mut retval,
            ));
            retval
        }
    }

    /// Updates the specific dimension.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of bounds or `s` is zero.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use dynet::Dim;
    /// let mut dim = Dim::from(([3, 4, 5], 8));
    /// assert_eq!(dim.get(2), 5);
    /// dim.set(2, 12);
    /// assert_eq!(dim.get(2), 12);
    /// ```
    ///
    /// A panic for out of bounds:
    ///
    /// ```should_panic
    /// # use dynet::Dim;
    /// let mut dim = Dim::from(([3, 4, 5], 8));
    /// dim.set(4, 12);
    /// ```
    pub fn set(&mut self, i: u32, s: u32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetDimDimensionSize(self.as_mut_ptr(), i, s));
        }
    }

    /// Trunsposes the dim.
    ///
    /// # Panics
    ///
    /// Panics if the dim has more than two dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dynet::Dim;
    /// let d1 = Dim::from([8]);
    /// let d1_transposed = d1.transpose();
    /// assert_eq!(d1_transposed.ndims(), 2);
    /// assert_eq!(d1_transposed, Dim::from([1, 8]));
    ///
    /// let d2 = Dim::from([4, 5]);
    /// let d2_transposed = d2.transpose();
    /// assert_eq!(d2_transposed, Dim::from([5, 4]));
    /// ```
    pub fn transpose(&self) -> Dim {
        unsafe {
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetTransposeDim(self.as_ptr(), &mut dim_ptr));
            Dim::from_raw(dim_ptr, true)
        }
    }
}

impl Default for Dim {
    fn default() -> Dim {
        Dim::new()
    }
}

impl PartialEq for Dim {
    fn eq(&self, other: &Dim) -> bool {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetIsDimEqualTo(
                self.as_ptr(),
                other.as_ptr(),
                &mut retval,
            ));
            retval == 1
        }
    }
}

impl Clone for Dim {
    #[inline]
    fn clone(&self) -> Dim {
        unsafe {
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCloneDim(self.as_ptr(), &mut dim_ptr));
            Dim::from_raw(dim_ptr, true)
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Dim) {
        unsafe {
            check_api_status!(dynet_sys::dynetDeleteDim(self.as_mut_ptr()));
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCloneDim(source.as_ptr(), &mut dim_ptr));
            self.inner = NonNull::new(dim_ptr).expect("pointer must not be null");
        }
    }
}

impl Eq for Dim {}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(dynet_sys::dynetRepresentDimAsString(
                self.as_ptr(),
                ptr::null_mut(),
                &mut size,
            ));
            let buffer = CString::new(vec![b'0'; size]).unwrap().into_raw();
            check_api_status!(dynet_sys::dynetRepresentDimAsString(
                self.as_ptr(),
                buffer,
                &mut size,
            ));
            f.write_str(CString::from_raw(buffer).to_str().unwrap())
        }
    }
}

macro_rules! impl_dim_from_array {
    ($num:expr) => {
        impl From<[u32; $num]> for Dim {
            fn from(dims: [u32; $num]) -> Dim {
                Dim::with_dimensions(&dims, 1)
            }
        }
    };
}
impl_dim_from_array!(0);
impl_dim_from_array!(1);
impl_dim_from_array!(2);
impl_dim_from_array!(3);
impl_dim_from_array!(4);
impl_dim_from_array!(5);
impl_dim_from_array!(6);
impl_dim_from_array!(7);
impl_dim_from_array!(8);

macro_rules! impl_dim_from_tuple {
    ($num:expr) => {
        impl From<([u32; $num], u32)> for Dim {
            fn from((dims, batch): ([u32; $num], u32)) -> Dim {
                Dim::with_dimensions(&dims, batch)
            }
        }
    };
}
impl_dim_from_tuple!(0);
impl_dim_from_tuple!(1);
impl_dim_from_tuple!(2);
impl_dim_from_tuple!(3);
impl_dim_from_tuple!(4);
impl_dim_from_tuple!(5);
impl_dim_from_tuple!(6);
impl_dim_from_tuple!(7);
impl_dim_from_tuple!(8);
