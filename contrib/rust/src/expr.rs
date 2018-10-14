use std::ptr::{self, NonNull};

use dynet_sys;

use super::{
    ApiResult, ComputationGraph, Device, Dim, LookupParameter, Parameter, Result, Tensor, Wrap,
};

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

macro_rules! expr_func_body {
    ($api_fn:ident, $($arg:expr),*) => {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::$api_fn(
                $($arg),*,
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }
}

/// Inputs scalar.
pub fn input_scalar(g: &mut ComputationGraph, s: f32) -> Expression {
    input_scalar_on(g, s, None)
}

/// Inputs scalar on the specified device.
pub fn input_scalar_on(
    g: &mut ComputationGraph,
    s: f32,
    device: Option<&mut Device>,
) -> Expression {
    expr_func_body!(
        dynetApplyInputScalar,
        g.as_mut_ptr(),
        s,
        device.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

/// Inputs vector/matrix/tensor.
pub fn input<D: Into<Dim>>(g: &mut ComputationGraph, d: D, data: &[f32]) -> Expression {
    input_on(g, d, data, None)
}

/// Inputs vector/matrix/tensor on the specified device.
pub fn input_on<D: Into<Dim>>(
    g: &mut ComputationGraph,
    d: D,
    data: &[f32],
    device: Option<&mut Device>,
) -> Expression {
    expr_func_body!(
        dynetApplyInput,
        g.as_mut_ptr(),
        d.into().as_ptr(),
        data.as_ptr(),
        data.len(),
        device.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

/// Inputs sparse vector.
pub fn input_sparse<D: Into<Dim>>(
    g: &mut ComputationGraph,
    d: D,
    ids: &[u32],
    data: &[f32],
    defdata: f32,
) -> Expression {
    input_sparse_on(g, d, ids, data, defdata, None)
}

/// Inputs sparse vector on the specified device.
///
/// # Arguments
///
/// * g - Computation graph.
/// * d - Dimension of the input matrix.
/// * ids - The indexes of the data points to update.
/// * data - The data points corresponding to each index.
/// * defdata - The default data with which to set the unspecified data points.
/// * device - The place device for the input value. If `None` is given, the default device will be
///            used instead.
pub fn input_sparse_on<D: Into<Dim>>(
    g: &mut ComputationGraph,
    d: D,
    ids: &[u32],
    data: &[f32],
    defdata: f32,
    device: Option<&mut Device>,
) -> Expression {
    expr_func_body!(
        dynetApplyInputSparse,
        g.as_mut_ptr(),
        d.into().as_ptr(),
        ids.as_ptr(),
        ids.len(),
        data.as_ptr(),
        data.len(),
        defdata,
        device.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

/// Creates batched one hot vectors on the specified device.
pub fn one_hot(g: &mut ComputationGraph, d: u32, ids: &[u32]) -> Expression {
    one_hot_on(g, d, ids, None)
}

/// Creates batched one hot vectors on the specified device.
///
/// # Arguments
///
/// * g - Computation graph.
/// * d - Dimension of the input vector.
/// * ids - The indices we want to set to 1, one per batch element.
/// * device - The place device for the input value. If `None` is given, the default device will be
///            used instead.
pub fn one_hot_on(
    g: &mut ComputationGraph,
    d: u32,
    ids: &[u32],
    device: Option<&mut Device>,
) -> Expression {
    expr_func_body!(
        dynetApplyOneHot,
        g.as_mut_ptr(),
        d,
        ids.as_ptr(),
        ids.len(),
        device.map(|d| d.as_mut_ptr()).unwrap_or(ptr::null_mut())
    )
}

/// Loads parameter.
pub fn parameter(g: &mut ComputationGraph, p: &mut Parameter) -> Expression {
    expr_func_body!(dynetApplyParameter, g.as_mut_ptr(), p.as_mut_ptr())
}

/// Loads lookup parameter.
pub fn lookup_parameter(g: &mut ComputationGraph, lp: &mut LookupParameter) -> Expression {
    expr_func_body!(dynetApplyLookupParameter, g.as_mut_ptr(), lp.as_mut_ptr())
}

/// Loads constant parameter.
pub fn const_parameter(g: &mut ComputationGraph, p: &Parameter) -> Expression {
    expr_func_body!(dynetApplyConstParameter, g.as_mut_ptr(), p.as_ptr())
}

/// Loads constant lookup parameter.
pub fn const_lookup_parameter(g: &mut ComputationGraph, lp: &LookupParameter) -> Expression {
    expr_func_body!(dynetApplyConstLookupParameter, g.as_mut_ptr(), lp.as_ptr())
}

/// Looks up parameter.
pub fn lookup_one(g: &mut ComputationGraph, p: &mut LookupParameter, index: u32) -> Expression {
    expr_func_body!(dynetApplyLookupOne, g.as_mut_ptr(), p.as_mut_ptr(), index)
}

/// Looks up parameters.
pub fn lookup(g: &mut ComputationGraph, p: &mut LookupParameter, indices: &[u32]) -> Expression {
    expr_func_body!(
        dynetApplyLookup,
        g.as_mut_ptr(),
        p.as_mut_ptr(),
        indices.as_ptr(),
        indices.len()
    )
}

/// Looks up constant parameter.
pub fn const_lookup_one(g: &mut ComputationGraph, p: &LookupParameter, index: u32) -> Expression {
    expr_func_body!(dynetApplyConstLookupOne, g.as_mut_ptr(), p.as_ptr(), index)
}

/// Looks up constant parameters.
pub fn const_lookup(g: &mut ComputationGraph, p: &LookupParameter, indices: &[u32]) -> Expression {
    expr_func_body!(
        dynetApplyConstLookup,
        g.as_mut_ptr(),
        p.as_ptr(),
        indices.as_ptr(),
        indices.len()
    )
}

/// Creates an input full of zeros.
pub fn zeros<D: Into<Dim>>(g: &mut ComputationGraph, d: D) -> Expression {
    expr_func_body!(dynetApplyZeros, g.as_mut_ptr(), d.into().as_ptr())
}

/// Creates an input full of ones.
pub fn ones<D: Into<Dim>>(g: &mut ComputationGraph, d: D) -> Expression {
    expr_func_body!(dynetApplyOnes, g.as_mut_ptr(), d.into().as_ptr())
}

/// Creates an input with one constant value.
pub fn constant<D: Into<Dim>>(g: &mut ComputationGraph, d: D, val: f32) -> Expression {
    expr_func_body!(dynetApplyConstant, g.as_mut_ptr(), d.into().as_ptr(), val)
}

/// Creates a random normal vector.
pub fn random_normal<D: Into<Dim>>(
    g: &mut ComputationGraph,
    d: D,
    mean: f32,
    stddev: f32,
) -> Expression {
    expr_func_body!(
        dynetApplyRandomNormal,
        g.as_mut_ptr(),
        d.into().as_ptr(),
        mean,
        stddev
    )
}

/// Creates a random bernoulli vector.
pub fn random_bernoulli<D: Into<Dim>>(
    g: &mut ComputationGraph,
    d: D,
    p: f32,
    scale: f32,
) -> Expression {
    expr_func_body!(
        dynetApplyRandomBernoulli,
        g.as_mut_ptr(),
        d.into().as_ptr(),
        p,
        scale
    )
}

/// Creates a random uniform vector.
pub fn random_uniform<D: Into<Dim>>(
    g: &mut ComputationGraph,
    d: D,
    left: f32,
    right: f32,
) -> Expression {
    expr_func_body!(
        dynetApplyRandomUniform,
        g.as_mut_ptr(),
        d.into().as_ptr(),
        left,
        right
    )
}

/// Creates a random gumbel vector.
pub fn random_gumbel<D: Into<Dim>>(
    g: &mut ComputationGraph,
    d: D,
    mu: f32,
    beta: f32,
) -> Expression {
    expr_func_body!(
        dynetApplyRandomGumbel,
        g.as_mut_ptr(),
        d.into().as_ptr(),
        mu,
        beta
    )
}
