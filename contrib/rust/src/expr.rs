use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, Dim, Result, Tensor, Wrap};

/// A struct to be the building block of a DyNet computation graph.
#[derive(Debug)]
pub struct Expression {
    inner: NonNull<dynet_sys::dynetExpression_t>,
}

impl_wrap_owned!(Expression, dynetExpression_t);
impl_drop!(Expression, dynetDeleteExpression);

impl Expression {
    /// Creates a new `Expression`.
    pub fn new() -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateExpression(&mut expr_ptr));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Returns the dimension of the expression.
    pub fn dim(&self) -> Dim {
        unsafe {
            let mut dim_ptr: *const dynet_sys::dynetDim_t = ptr::null();
            check_api_status!(dynet_sys::dynetGetExpressionDim(
                self.as_ptr(),
                &mut dim_ptr
            ));
            Dim::from_raw(dim_ptr as *mut _, false).clone()
        }
    }

    /// Returns the value of the expression.
    pub fn value(&mut self) -> Tensor {
        unsafe {
            let mut tensor_ptr: *const dynet_sys::dynetTensor_t = ptr::null();
            check_api_status!(dynet_sys::dynetGetExpressionValue(
                self.as_mut_ptr(),
                &mut tensor_ptr,
            ));
            Tensor::from_raw(tensor_ptr as *mut _, false)
        }
    }

    /// Returns the gradient of the expression.
    pub fn gradient(&mut self) -> Tensor {
        unsafe {
            let mut tensor_ptr: *const dynet_sys::dynetTensor_t = ptr::null();
            check_api_status!(dynet_sys::dynetGetExpressionGradient(
                self.as_mut_ptr(),
                &mut tensor_ptr,
            ));
            Tensor::from_raw(tensor_ptr as *mut _, false)
        }
    }
}

impl Default for Expression {
    fn default() -> Expression {
        Expression::new()
    }
}

impl AsRef<Expression> for Expression {
    #[inline]
    fn as_ref(&self) -> &Expression {
        self
    }
}
