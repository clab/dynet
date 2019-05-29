use std::ops;
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
            let mut src_ptr: *const dynet_sys::dynetDim_t = ptr::null();
            check_api_status!(dynet_sys::dynetGetExpressionDim(
                self.as_ptr(),
                &mut src_ptr
            ));
            let mut dim_ptr: *mut dynet_sys::dynetDim_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCloneDim(src_ptr, &mut dim_ptr));
            Dim::from_raw(dim_ptr, true)
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

impl Clone for Expression {
    #[inline]
    fn clone(&self) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCloneExpression(
                self.as_ptr(),
                &mut expr_ptr
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Expression) {
        unsafe {
            check_api_status!(dynet_sys::dynetDeleteExpression(self.as_mut_ptr()));
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCloneExpression(
                source.as_ptr(),
                &mut expr_ptr
            ));
            self.inner = NonNull::new(expr_ptr).expect("pointer must not be null");
        }
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

macro_rules! impl_expr_unary_func {
    ($name:ident, $api_fn:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name<E: AsRef<Expression>>(x: E) -> Expression {
            expr_func_body!($api_fn, x.as_ref().as_ptr())
        }
    };
}

impl_expr_unary_func!(negative, dynetApplyNegative, "Applies negation operation.");

impl ops::Neg for Expression {
    type Output = Expression;

    fn neg(self) -> Expression {
        expr_func_body!(dynetApplyNegative, self.as_ptr())
    }
}

macro_rules! impl_expr_binary_func {
    (
        $name:ident,
        $api_fn:ident,
        $name_xc:ident,
        $api_fn_xc:ident,
        $name_cx:ident,
        $api_fn_cx:ident,
        $doc:expr
    ) => {
        impl_expr_binary_func!($name, $api_fn, $name_xc, $api_fn_xc, $doc);

        #[doc = $doc]
        pub fn $name_cx<E: AsRef<Expression>>(x: f32, y: E) -> Expression {
            expr_func_body!($api_fn_cx, x, y.as_ref().as_ptr())
        }
    };
    ($name:ident, $api_fn:ident, $name_xc:ident, $api_fn_xc:ident, $doc:expr) => {
        impl_expr_binary_func!($name, $api_fn, $doc);

        #[doc = $doc]
        pub fn $name_xc<E: AsRef<Expression>>(x: E, y: f32) -> Expression {
            expr_func_body!($api_fn_xc, x.as_ref().as_ptr(), y)
        }
    };
    ($name:ident, $api_fn:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name<E1: AsRef<Expression>, E2: AsRef<Expression>>(x: E1, y: E2) -> Expression {
            expr_func_body!($api_fn, x.as_ref().as_ptr(), y.as_ref().as_ptr())
        }
    };
}

macro_rules! impl_expr_binary_with_constant_op {
    ($scalar:ty, $name:ident, $op_fn:ident, $api_fn_xc:ident, $api_fn_cx:ident) => {
        impl_expr_binary_with_constant_op!($scalar, $name, $op_fn, $api_fn_xc);

        impl ops::$name<Expression> for $scalar {
            type Output = Expression;

            #[allow(trivial_numeric_casts)]
            fn $op_fn(self, rhs: Expression) -> Expression {
                expr_func_body!($api_fn_cx, self as f32, rhs.as_ptr())
            }
        }

        impl<'a> ops::$name<&'a Expression> for $scalar {
            type Output = Expression;

            #[allow(trivial_numeric_casts)]
            fn $op_fn(self, rhs: &'a Expression) -> Expression {
                expr_func_body!($api_fn_cx, self as f32, rhs.as_ptr())
            }
        }
    };
    ($scalar:ty, $name:ident, $op_fn:ident, $api_fn_xc:ident) => {
        impl ops::$name<$scalar> for Expression {
            type Output = Expression;

            #[allow(trivial_numeric_casts)]
            fn $op_fn(self, rhs: $scalar) -> Expression {
                expr_func_body!($api_fn_xc, self.as_ptr(), rhs as f32)
            }
        }

        impl<'a> ops::$name<$scalar> for &'a Expression {
            type Output = Expression;

            #[allow(trivial_numeric_casts)]
            fn $op_fn(self, rhs: $scalar) -> Expression {
                expr_func_body!($api_fn_xc, self.as_ptr(), rhs as f32)
            }
        }
    };
}

macro_rules! impl_expr_binary_op {
    ($name:ident, $op_fn:ident, $api_fn:ident, $api_fn_xc:ident, $api_fn_cx:ident) => {
        impl_expr_binary_with_constant_op!(i8, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_expr_binary_with_constant_op!(u8, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_expr_binary_with_constant_op!(i16, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_expr_binary_with_constant_op!(u16, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_expr_binary_with_constant_op!(i32, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_expr_binary_with_constant_op!(u32, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_expr_binary_with_constant_op!(i64, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_expr_binary_with_constant_op!(u64, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_expr_binary_with_constant_op!(f32, $name, $op_fn, $api_fn_xc, $api_fn_cx);
        impl_expr_binary_with_constant_op!(f64, $name, $op_fn, $api_fn_xc, $api_fn_cx);

        impl_expr_binary_op!($name, $op_fn, $api_fn);
    };
    ($name:ident, $op_fn:ident, $api_fn:ident, $api_fn_xc:ident) => {
        impl_expr_binary_with_constant_op!(i8, $name, $op_fn, $api_fn_xc);
        impl_expr_binary_with_constant_op!(u8, $name, $op_fn, $api_fn_xc);
        impl_expr_binary_with_constant_op!(i16, $name, $op_fn, $api_fn_xc);
        impl_expr_binary_with_constant_op!(u16, $name, $op_fn, $api_fn_xc);
        impl_expr_binary_with_constant_op!(i32, $name, $op_fn, $api_fn_xc);
        impl_expr_binary_with_constant_op!(u32, $name, $op_fn, $api_fn_xc);
        impl_expr_binary_with_constant_op!(i64, $name, $op_fn, $api_fn_xc);
        impl_expr_binary_with_constant_op!(u64, $name, $op_fn, $api_fn_xc);
        impl_expr_binary_with_constant_op!(f32, $name, $op_fn, $api_fn_xc);
        impl_expr_binary_with_constant_op!(f64, $name, $op_fn, $api_fn_xc);

        impl_expr_binary_op!($name, $op_fn, $api_fn);
    };
    ($name:ident, $op_fn:ident, $api_fn:ident) => {
        impl ops::$name for Expression {
            type Output = Expression;

            fn $op_fn(self, rhs: Expression) -> Expression {
                expr_func_body!($api_fn, self.as_ptr(), rhs.as_ptr())
            }
        }

        impl<'a> ops::$name<Expression> for &'a Expression {
            type Output = Expression;

            fn $op_fn(self, rhs: Expression) -> Expression {
                expr_func_body!($api_fn, self.as_ptr(), rhs.as_ptr())
            }
        }

        impl<'a> ops::$name<&'a Expression> for Expression {
            type Output = Expression;

            fn $op_fn(self, rhs: &'a Expression) -> Expression {
                expr_func_body!($api_fn, self.as_ptr(), rhs.as_ptr())
            }
        }

        impl<'a, 'b> ops::$name<&'a Expression> for &'b Expression {
            type Output = Expression;

            fn $op_fn(self, rhs: &'a Expression) -> Expression {
                expr_func_body!($api_fn, self.as_ptr(), rhs.as_ptr())
            }
        }
    };
}

impl_expr_binary_func!(
    add,
    dynetApplyAdd,
    add_const,
    dynetApplyAddConst,
    add_expr,
    dynetApplyAddExpr,
    "Applies addition operation."
);
impl_expr_binary_op!(
    Add,
    add,
    dynetApplyAdd,
    dynetApplyAddConst,
    dynetApplyAddExpr
);
impl_expr_binary_func!(
    subtract,
    dynetApplySubtract,
    subtract_const,
    dynetApplySubtractConst,
    subtract_expr,
    dynetApplySubtractExpr,
    "Applies subtraction operation."
);
impl_expr_binary_op!(
    Sub,
    sub,
    dynetApplySubtract,
    dynetApplySubtractConst,
    dynetApplySubtractExpr
);
impl_expr_binary_func!(
    multiply,
    dynetApplyMultiply,
    multiply_const,
    dynetApplyMultiplyConst,
    multiply_expr,
    dynetApplyMultiplyExpr,
    "Applies multiplication operation."
);
impl_expr_binary_op!(
    Mul,
    mul,
    dynetApplyMultiply,
    dynetApplyMultiplyConst,
    dynetApplyMultiplyExpr
);
impl_expr_binary_func!(
    divide,
    dynetApplyDivide,
    divide_const,
    dynetApplyDivideConst,
    "Applies division operation."
);
impl_expr_binary_op!(Div, div, dynetApplyDivide, dynetApplyDivideConst);

macro_rules! impl_expr_nary_func {
    ($name:ident, $api_fn:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name<ES: AsRef<[E]>, E: AsRef<Expression>>(xs: ES) -> Expression {
            let x_ptrs: Vec<_> = xs.as_ref().iter().map(|x| x.as_ref().as_ptr()).collect();
            expr_func_body!($api_fn, x_ptrs.as_ptr(), x_ptrs.len())
        }
    };
}

impl_expr_nary_func!(
    affine_transform,
    dynetApplyAffineTransform,
    "
Applies affine transform operation.\n
\n
This performs an affine transform over an arbitrary (odd) number of expressions held in the input
initializer list xs. The first expression is the \"bias,\" which is added to the expression as-is.
The remaining expressions are multiplied together in pairs, then added. A very common usage case is
the calculation of the score for a neural network layer (e.g. b + Wz) where b is the bias, W is the
weight matrix, and z is the input. In this case xs[0] = b, xs[1] = W, and xs[2] = z."
);
impl_expr_nary_func!(
    sum,
    dynetApplySum,
    "
Applies sum operation.\n 
\n
This returns an expression where the ith element is equal to xs[0][i] + xs[1][i] + ... ."
);

impl_expr_unary_func!(sum_elems, dynetApplySumElems, "Sums all elements.");

/// Computes moment over all elements.
///
/// # Arguments
///
/// * x - Input mini-batched expression.
/// * r - Order of the moment.
pub fn moment_elems<E: AsRef<Expression>>(x: E, r: u32) -> Expression {
    expr_func_body!(dynetApplyMomentElems, x.as_ref().as_ptr(), r)
}

impl_expr_unary_func!(
    mean_elems,
    dynetApplyMeanElems,
    "Computes mean over all elements."
);
impl_expr_unary_func!(
    std_elems,
    dynetApplyStdElems,
    "Computes standard deviation over all elements."
);
impl_expr_unary_func!(sum_batches, dynetApplySumBatches, "Sums up mini-batches.");

/// Computes moment over mini-batches.
///
/// # Arguments
///
/// * x - Input mini-batched expression.
/// * r - Order of the moment.
pub fn moment_batches<E: AsRef<Expression>>(x: E, r: u32) -> Expression {
    expr_func_body!(dynetApplyMomentBatches, x.as_ref().as_ptr(), r)
}

impl_expr_unary_func!(
    mean_batches,
    dynetApplyMeanBatches,
    "Computes mean over over mini-batches."
);
impl_expr_unary_func!(
    std_batches,
    dynetApplyStdBatches,
    "Computes standard deviation over over mini-batches."
);

/// Computes sum along a specific dimension(s).
///
/// # Arguments
///
/// * x - Input mini-batched expression.
/// * dims - Dimensions along which to reduce.
/// * b - Whether to include batch dimension.
pub fn sum_dim<E: AsRef<Expression>>(x: E, dims: &[u32], b: bool) -> Expression {
    expr_func_body!(
        dynetApplySumDim,
        x.as_ref().as_ptr(),
        dims.as_ptr(),
        dims.len(),
        b as u32
    )
}

/// Computes cumulative sum along a specific dimension.
///
/// # Arguments
///
/// * x - Input mini-batched expression.
/// * d - Dimension along which to compute the cumulative sum.
pub fn cumsum<E: AsRef<Expression>>(x: E, d: u32) -> Expression {
    expr_func_body!(dynetApplyCumsum, x.as_ref().as_ptr(), d)
}

/// Computes moment along a specific dimension.
///
/// # Arguments
///
/// * x - Input mini-batched expression.
/// * dims - Dimensions along which to reduce.
/// * r - Order of the moment.
/// * b - Whether to include batch dimension.
/// * n - If > 0, overwrite the `n` in the equation by this value, useful for masking.
pub fn moment_dim<E: AsRef<Expression>>(x: E, dims: &[u32], r: u32, b: bool, n: u32) -> Expression {
    expr_func_body!(
        dynetApplyMomentDim,
        x.as_ref().as_ptr(),
        dims.as_ptr(),
        dims.len(),
        r,
        b as u32,
        n
    )
}

/// Computes mean along a specific dimension.
///
/// # Arguments
///
/// * x - Input mini-batched expression.
/// * dims - Dimensions along which to reduce.
/// * b - Whether to include batch dimension.
/// * n - If > 0, overwrite the `n` in the equation by this value, useful for masking.
pub fn mean_dim<E: AsRef<Expression>>(x: E, dims: &[u32], b: bool, n: u32) -> Expression {
    expr_func_body!(
        dynetApplyMeanDim,
        x.as_ref().as_ptr(),
        dims.as_ptr(),
        dims.len(),
        b as u32,
        n
    )
}

/// Computes standard deviation along a specific dimension.
///
/// # Arguments
///
/// * x - Input mini-batched expression.
/// * dims - Dimensions along which to reduce.
/// * b - Whether to include batch dimension.
/// * n - If > 0, overwrite the `n` in the equation by this value, useful for masking.
pub fn std_dim<E: AsRef<Expression>>(x: E, dims: &[u32], b: bool, n: u32) -> Expression {
    expr_func_body!(
        dynetApplyStdDim,
        x.as_ref().as_ptr(),
        dims.as_ptr(),
        dims.len(),
        b as u32,
        n
    )
}

impl_expr_nary_func!(
    average,
    dynetApplyAverage,
    "Computes element-wise average over all expressions."
);
impl_expr_unary_func!(sqrt, dynetApplySqrt, "Computes square root.");
impl_expr_unary_func!(abs, dynetApplyAbs, "Computes absolute value.");
impl_expr_unary_func!(
    erf,
    dynetApplyErf,
    "Computes the value of the Gaussian error function."
);
impl_expr_unary_func!(asin, dynetApplyAsin, "Computes inverse sine.");
impl_expr_unary_func!(acos, dynetApplyAcos, "Computes inverse cosine.");
impl_expr_unary_func!(atan, dynetApplyAtan, "Computes inverse tangent.");
impl_expr_unary_func!(sin, dynetApplySin, "Computes sine.");
impl_expr_unary_func!(cos, dynetApplyCos, "Computes cosine.");
impl_expr_unary_func!(tan, dynetApplyTan, "Computes tangent.");
impl_expr_unary_func!(sinh, dynetApplySinh, "Computes hyperbolic sine.");
impl_expr_unary_func!(cosh, dynetApplyCosh, "Computes hyperbolic cosine.");
impl_expr_unary_func!(tanh, dynetApplyTanh, "Computes hyperbolic tangent.");
impl_expr_unary_func!(asinh, dynetApplyAsinh, "Computes inverse hyperbolic sine.");
impl_expr_unary_func!(
    acosh,
    dynetApplyAcosh,
    "Computes inverse hyperbolic cosine."
);
impl_expr_unary_func!(
    atanh,
    dynetApplyAtanh,
    "Computes inverse hyperbolic tangent."
);
impl_expr_unary_func!(exp, dynetApplyExp, "Computes natural exponent.");
impl_expr_unary_func!(square, dynetApplySquare, "Computes square.");
impl_expr_unary_func!(cube, dynetApplyCube, "Computes cube.");
impl_expr_unary_func!(log_sigmoid, dynetApplyLogSigmoid, "Computes log sigmoid.");
impl_expr_unary_func!(lgamma, dynetApplyLgamma, "Computes log gamma.");
impl_expr_unary_func!(log, dynetApplyLog, "Computes logarithm.");
impl_expr_unary_func!(logistic, dynetApplyLogistic, "Computes logistic sigmoid.");
impl_expr_unary_func!(rectify, dynetApplyRectify, "Computes rectifier.");

/// Computes exponential linear unit.
pub fn elu<E: AsRef<Expression>>(x: E, alpha: f32) -> Expression {
    expr_func_body!(dynetApplyElu, x.as_ref().as_ptr(), alpha)
}

impl_expr_unary_func!(
    selu,
    dynetApplySelu,
    "Computes scaled exponential linear unit."
);

/// Computes SILU / SiL / Swish.
pub fn silu<E: AsRef<Expression>>(x: E, beta: f32) -> Expression {
    expr_func_body!(dynetApplySilu, x.as_ref().as_ptr(), beta)
}

impl_expr_unary_func!(softsign, dynetApplySoftsign, "Computes soft sign.");
impl_expr_binary_func!(pow, dynetApplyPow, "Computes power.");
impl_expr_binary_func!(bmin, dynetApplyBmin, "Computes binary minimum.");
impl_expr_binary_func!(bmax, dynetApplyBmax, "Computes binary maximum.");
impl_expr_nary_func!(max, dynetApplyMax, "Computes maximum over all expressions.");
impl_expr_binary_func!(dot_product, dynetApplyDotProduct, "Computes dot product.");

/// Computes circular convolution.
pub fn circ_conv<E1: AsRef<Expression>, E2: AsRef<Expression>>(u: E1, v: E2) -> Expression {
    expr_func_body!(dynetApplyCircConv, u.as_ref().as_ptr(), v.as_ref().as_ptr())
}

/// Computes circular correlation.
pub fn circ_corr<E1: AsRef<Expression>, E2: AsRef<Expression>>(u: E1, v: E2) -> Expression {
    expr_func_body!(dynetApplyCircCorr, u.as_ref().as_ptr(), v.as_ref().as_ptr())
}

impl_expr_binary_func!(
    cmult,
    dynetApplyCmult,
    "Computes componentwise multiplication."
);
impl_expr_binary_func!(cdiv, dynetApplyCdiv, "Computes componentwise division.");

/// Computes columnwise addition.
pub fn colwise_add<E1: AsRef<Expression>, E2: AsRef<Expression>>(x: E1, bias: E2) -> Expression {
    expr_func_body!(
        dynetApplyColwiseAdd,
        x.as_ref().as_ptr(),
        bias.as_ref().as_ptr()
    )
}

/// Computes rounding
pub fn round<E: AsRef<Expression>>(x: E) -> Expression {
    round_with_zero_gradient_mode(x)
}
impl_expr_unary_func!(
    round_with_zero_gradient_mode,
    dynetApplyRoundWithZeroGradientMode,
    "Computes rounding with zero gradient mode"
);
impl_expr_unary_func!(
    round_with_straight_through_gradient_mode,
    dynetApplyRoundWithStraightThroughGradientMode,
    "Computes rounding with straight through gradient mode"
);

/// Computes ceiling
pub fn ceil<E: AsRef<Expression>>(x: E) -> Expression {
    ceil_with_zero_gradient_mode(x)
}
impl_expr_unary_func!(
    ceil_with_zero_gradient_mode,
    dynetApplyCeilWithZeroGradientMode,
    "Computes ceiling with zero gradient mode"
);
impl_expr_unary_func!(
    ceil_with_straight_through_gradient_mode,
    dynetApplyCeilWithStraightThroughGradientMode,
    "Computes ceiling with straight through gradient mode"
);

/// Computes floor
pub fn floor<E: AsRef<Expression>>(x: E) -> Expression {
    floor_with_zero_gradient_mode(x)
}
impl_expr_unary_func!(
    floor_with_zero_gradient_mode,
    dynetApplyFloorWithZeroGradientMode,
    "Computes floor with zero gradient mode"
);
impl_expr_unary_func!(
    floor_with_straight_through_gradient_mode,
    dynetApplyFloorWithStraightThroughGradientMode,
    "Computes floor with straight through gradient mode"
);

/// Computes softmax
pub fn softmax<E: AsRef<Expression>>(x: E, d: u32) -> Expression {
    expr_func_body!(dynetApplySoftmax, x.as_ref().as_ptr(), d)
}

impl_expr_unary_func!(log_softmax, dynetApplyLogSoftmax, "Computes log softmax");

/// Computes restricted log softmax
pub fn restricted_log_softmax<E: AsRef<Expression>>(x: E, restriction: &[u32]) -> Expression {
    expr_func_body!(
        dynetApplyRestrictedLogSoftmax,
        x.as_ref().as_ptr(),
        restriction.as_ptr(),
        restriction.len()
    )
}

/// Computes log, sum, and exp by dimension
pub fn logsumexp_dim<E: AsRef<Expression>>(x: E, d: u32) -> Expression {
    expr_func_body!(dynetApplyLogsumexpDim, x.as_ref().as_ptr(), d)
}

impl_expr_nary_func!(logsumexp, dynetApplyLogsumexp, "Computes log, sum, and exp");

/// Computes negative softmax log likelihood
pub fn pickneglogsoftmax_one<E: AsRef<Expression>>(x: E, v: u32) -> Expression {
    expr_func_body!(dynetApplyPickneglogsoftmaxOne, x.as_ref().as_ptr(), v)
}

/// Computes batched negative softmax log likelihood
pub fn pickneglogsoftmax<E: AsRef<Expression>>(x: E, v: &[u32]) -> Expression {
    expr_func_body!(
        dynetApplyPickneglogsoftmax,
        x.as_ref().as_ptr(),
        v.as_ptr(),
        v.len()
    )
}

/// Computes hinge loss
pub fn hinge_one<E: AsRef<Expression>>(x: E, index: u32, m: f32) -> Expression {
    expr_func_body!(dynetApplyHingeOne, x.as_ref().as_ptr(), index, m)
}

/// Computes batched hinge loss
pub fn hinge<E: AsRef<Expression>>(x: E, indices: &[u32], m: f32) -> Expression {
    expr_func_body!(
        dynetApplyHinge,
        x.as_ref().as_ptr(),
        indices.as_ptr(),
        indices.len(),
        m
    )
}

/// Computes dimensionwise hinge loss
pub fn hinge_dim_one<E: AsRef<Expression>>(x: E, indices: &[u32], d: u32, m: f32) -> Expression {
    expr_func_body!(
        dynetApplyHingeDimOne,
        x.as_ref().as_ptr(),
        indices.as_ptr(),
        indices.len(),
        d,
        m
    )
}

/// Computes batched dimensionwise hinge loss
pub fn hinge_dim<E: AsRef<Expression>>(x: E, indices: &[u32], d: u32, m: f32) -> Expression {
    expr_func_body!(
        dynetApplyHingeDim,
        x.as_ref().as_ptr(),
        indices.as_ptr(),
        indices.len(),
        d,
        m
    )
}

impl_expr_unary_func!(sparsemax, dynetApplySparsemax, "Computes sparsemax");

/// Computes sparsemax loss
pub fn sparsemax_loss<E: AsRef<Expression>>(x: E, target_support: &[u32]) -> Expression {
    expr_func_body!(
        dynetApplySparsemaxLoss,
        x.as_ref().as_ptr(),
        target_support.as_ptr(),
        target_support.len()
    )
}

impl_expr_binary_func!(
    constrained_softmax,
    dynetApplyConstrainedSoftmax,
    "Computes constrained softmax"
);
impl_expr_unary_func!(squared_norm, dynetApplySquaredNorm, "Computes squared norm");
impl_expr_unary_func!(l2_norm, dynetApplyL2Norm, "Computes L2 norm");
impl_expr_binary_func!(
    squared_distance,
    dynetApplySquaredDistance,
    "Computes squared distance"
);
impl_expr_binary_func!(l1_distance, dynetApplyL1Distance, "Computes L1 distance");

/// Computes huber distance
pub fn huber_distance<E1: AsRef<Expression>, E2: AsRef<Expression>>(
    x: E1,
    y: E2,
    c: f32,
) -> Expression {
    expr_func_body!(
        dynetApplyHuberDistance,
        x.as_ref().as_ptr(),
        y.as_ref().as_ptr(),
        c
    )
}

impl_expr_binary_func!(
    binary_log_loss,
    dynetApplyBinaryLogLoss,
    "Computes binary log loss"
);

/// Computes pairwise rank loss
pub fn pairwise_rank_loss<E1: AsRef<Expression>, E2: AsRef<Expression>>(
    x: E1,
    y: E2,
    m: f32,
) -> Expression {
    expr_func_body!(
        dynetApplyPairwiseRankLoss,
        x.as_ref().as_ptr(),
        y.as_ref().as_ptr(),
        m
    )
}

/// Computes Poisson loss
pub fn poisson_loss<E: AsRef<Expression>>(x: E, y: u32) -> Expression {
    expr_func_body!(dynetApplyPoissonLoss, x.as_ref().as_ptr(), y)
}

impl_expr_unary_func!(nobackporp, dynetApplyNobackprop, "Prevents backprop");
impl_expr_unary_func!(flip_gradient, dynetApplyFlipGradient, "Flips gradient");

/// Scales gradient by constant
pub fn scale_gradient<E: AsRef<Expression>>(x: E, lambd: f32) -> Expression {
    expr_func_body!(dynetApplyScaleGradient, x.as_ref().as_ptr(), lambd)
}

/// Computes argmax
pub fn argmax<E: AsRef<Expression>>(x: E) -> Expression {
    argmax_with_zero_gradient_mode(x)
}
impl_expr_unary_func!(
    argmax_with_zero_gradient_mode,
    dynetApplyArgmaxWithZeroGradientMode,
    "Computes argmax with zero gradient mode"
);
impl_expr_unary_func!(
    argmax_with_straight_through_gradient_mode,
    dynetApplyArgmaxWithStraightThroughGradientMode,
    "Computes argmax with straight through gradient mode"
);

/// Reshapes to another size
pub fn reshape<E: AsRef<Expression>, D: Into<Dim>>(x: E, d: D) -> Expression {
    expr_func_body!(dynetApplyReshape, x.as_ref().as_ptr(), d.into().as_ptr())
}

/// Transposes a matrix
pub fn transpose<E: AsRef<Expression>>(x: E, dims: &[u32]) -> Expression {
    expr_func_body!(
        dynetApplyTranspose,
        x.as_ref().as_ptr(),
        dims.as_ptr(),
        dims.len()
    )
}

/// Selects rows
pub fn select_rows<E: AsRef<Expression>>(x: E, rows: &[u32]) -> Expression {
    expr_func_body!(
        dynetApplySelectRows,
        x.as_ref().as_ptr(),
        rows.as_ptr(),
        rows.len()
    )
}

/// Selects cols
pub fn select_cols<E: AsRef<Expression>>(x: E, cols: &[u32]) -> Expression {
    expr_func_body!(
        dynetApplySelectCols,
        x.as_ref().as_ptr(),
        cols.as_ptr(),
        cols.len()
    )
}

/// Picks element
pub fn pick_one<E: AsRef<Expression>>(x: E, v: u32, d: u32) -> Expression {
    expr_func_body!(dynetApplyPickOne, x.as_ref().as_ptr(), v, d)
}

/// Picks elements from batches
pub fn pick<E: AsRef<Expression>>(x: E, v: &[u32], d: u32) -> Expression {
    expr_func_body!(dynetApplyPick, x.as_ref().as_ptr(), v.as_ptr(), v.len(), d)
}

/// Picks range of elements
pub fn pick_range<E: AsRef<Expression>>(x: E, s: u32, e: u32, d: u32) -> Expression {
    expr_func_body!(dynetApplyPickRange, x.as_ref().as_ptr(), s, e, d)
}

/// Picks batch element
pub fn pick_batch_elem<E: AsRef<Expression>>(x: E, v: u32) -> Expression {
    expr_func_body!(dynetApplyPickBatchElem, x.as_ref().as_ptr(), v)
}

/// Picks batch elements
pub fn pick_batch_elems<E: AsRef<Expression>>(x: E, v: &[u32]) -> Expression {
    expr_func_body!(
        dynetApplyPickBatchElems,
        x.as_ref().as_ptr(),
        v.as_ptr(),
        v.len()
    )
}

/// Stridingly selects in multiple dimensions
pub fn strided_select<E: AsRef<Expression>>(
    x: E,
    strides: &[i32],
    from: &[i32],
    to: &[i32],
) -> Expression {
    expr_func_body!(
        dynetApplyStridedSelect,
        x.as_ref().as_ptr(),
        strides.as_ptr(),
        strides.len(),
        from.as_ptr(),
        from.len(),
        to.as_ptr(),
        to.len()
    )
}

impl_expr_nary_func!(
    concatenate_to_batch,
    dynetApplyConcatenateToBatch,
    "Concatenates list of expressions to a single batched expression"
);
impl_expr_nary_func!(
    concatenate_cols,
    dynetApplyConcatenateCols,
    "Concatenates columns"
);

/// Concatenates expressions
pub fn concatenate<ES: AsRef<[E]>, E: AsRef<Expression>>(xs: ES, d: u32) -> Expression {
    let x_ptrs: Vec<_> = xs.as_ref().iter().map(|x| x.as_ref().as_ptr()).collect();
    expr_func_body!(dynetApplyConcatenate, x_ptrs.as_ptr(), x_ptrs.len(), d)
}

/// Selects max out through a dimension
pub fn max_dim<E: AsRef<Expression>>(x: E, d: u32) -> Expression {
    expr_func_body!(dynetApplyMaxDim, x.as_ref().as_ptr(), d)
}

/// Selects min out through a dimension
pub fn min_dim<E: AsRef<Expression>>(x: E, d: u32) -> Expression {
    expr_func_body!(dynetApplyMinDim, x.as_ref().as_ptr(), d)
}

/// Adds Gaussian noise
pub fn noise<E: AsRef<Expression>>(x: E, stddev: f32) -> Expression {
    expr_func_body!(dynetApplyNoise, x.as_ref().as_ptr(), stddev)
}

/// Applies dropout
pub fn dropout<E: AsRef<Expression>>(x: E, p: f32) -> Expression {
    expr_func_body!(dynetApplyDropout, x.as_ref().as_ptr(), p)
}

/// Applies dropout along a specific dimension
pub fn dropout_dim<E: AsRef<Expression>>(x: E, d: u32, p: f32) -> Expression {
    expr_func_body!(dynetApplyDropoutDim, x.as_ref().as_ptr(), d, p)
}

/// Applies dropout to entire elements of a minibatch
pub fn dropout_batch<E: AsRef<Expression>>(x: E, p: f32) -> Expression {
    expr_func_body!(dynetApplyDropoutBatch, x.as_ref().as_ptr(), p)
}

/// Applies block dropout
pub fn block_dropout<E: AsRef<Expression>>(x: E, p: f32) -> Expression {
    expr_func_body!(dynetApplyBlockDropout, x.as_ref().as_ptr(), p)
}

/// Convolution operation
pub fn filter1d_narrow<E1: AsRef<Expression>, E2: AsRef<Expression>>(x: E1, f: E2) -> Expression {
    expr_func_body!(
        dynetApplyFilter1dNarrow,
        x.as_ref().as_ptr(),
        f.as_ref().as_ptr()
    )
}

/// Selects out k maximum values along a given dimension
pub fn kmax_pooling<E: AsRef<Expression>>(x: E, k: u32, d: u32) -> Expression {
    expr_func_body!(dynetApplyKmaxPooling, x.as_ref().as_ptr(), k, d)
}

/// Convolution operation
pub fn fold_rows<E: AsRef<Expression>>(x: E, nrows: u32) -> Expression {
    expr_func_body!(dynetApplyFoldRows, x.as_ref().as_ptr(), nrows)
}

impl_expr_unary_func!(average_cols, dynetApplyAverageCols, "Convolution operation");

/// Convolution operation
pub fn kmh_ngram<E: AsRef<Expression>>(x: E, n: u32) -> Expression {
    expr_func_body!(dynetApplyKmhNgram, x.as_ref().as_ptr(), n)
}

/// Applies 2D convolution operation without bias parameters
pub fn conv2d<E1: AsRef<Expression>, E2: AsRef<Expression>>(
    x: E1,
    f: E2,
    stride: &[u32],
    is_valid: bool,
) -> Expression {
    expr_func_body!(
        dynetApplyConv2d,
        x.as_ref().as_ptr(),
        f.as_ref().as_ptr(),
        stride.as_ptr(),
        stride.len(),
        is_valid as u32
    )
}

/// Applies 2D convolution operation with bias parameters
pub fn conv2d_with_bias<E1: AsRef<Expression>, E2: AsRef<Expression>, E3: AsRef<Expression>>(
    x: E1,
    f: E2,
    b: E3,
    stride: &[u32],
    is_valid: bool,
) -> Expression {
    expr_func_body!(
        dynetApplyConv2dWithBias,
        x.as_ref().as_ptr(),
        f.as_ref().as_ptr(),
        b.as_ref().as_ptr(),
        stride.as_ptr(),
        stride.len(),
        is_valid as u32
    )
}

/// Applies 2D maxpooling operation
pub fn maxpooling2d<E: AsRef<Expression>>(
    x: E,
    ksize: &[u32],
    stride: &[u32],
    is_valid: bool,
) -> Expression {
    expr_func_body!(
        dynetApplyMaxpooling2d,
        x.as_ref().as_ptr(),
        ksize.as_ptr(),
        ksize.len(),
        stride.as_ptr(),
        stride.len(),
        is_valid as u32
    )
}

impl_expr_binary_func!(
    contract3d_1d,
    dynetApplyContract3d1d,
    "Contracts a rank 3 tensor and a rank 1 tensor into a rank 2 tensor"
);

/// Contracts a rank 3 tensor and a rank 1 tensor into a rank 2 tensor with an additional bias
/// parameter
pub fn contract3d_1d_with_bias<
    E1: AsRef<Expression>,
    E2: AsRef<Expression>,
    E3: AsRef<Expression>,
>(
    x: E1,
    y: E2,
    b: E3,
) -> Expression {
    expr_func_body!(
        dynetApplyContract3d1dWithBias,
        x.as_ref().as_ptr(),
        y.as_ref().as_ptr(),
        b.as_ref().as_ptr()
    )
}

/// Contracts a rank 3 tensor and two rank 1 tensor into a rank 1 tensor
pub fn contract3d_1d_1d<E1: AsRef<Expression>, E2: AsRef<Expression>, E3: AsRef<Expression>>(
    x: E1,
    y: E2,
    z: E3,
) -> Expression {
    expr_func_body!(
        dynetApplyContract3d1d1d,
        x.as_ref().as_ptr(),
        y.as_ref().as_ptr(),
        z.as_ref().as_ptr()
    )
}

/// Contracts a rank 3 tensor and two rank 1 tensor into a rank 1 tensor with an additional bias
/// parameter
pub fn contract3d_1d_1d_with_bias<
    E1: AsRef<Expression>,
    E2: AsRef<Expression>,
    E3: AsRef<Expression>,
    E4: AsRef<Expression>,
>(
    x: E1,
    y: E2,
    z: E3,
    b: E3,
) -> Expression {
    expr_func_body!(
        dynetApplyContract3d1d1dWithBias,
        x.as_ref().as_ptr(),
        y.as_ref().as_ptr(),
        z.as_ref().as_ptr(),
        b.as_ref().as_ptr()
    )
}

impl_expr_unary_func!(inverse, dynetApplyInverse, "Takes the inverse of a matrix");
impl_expr_unary_func!(
    logdet,
    dynetApplyLogdet,
    "Takes the log of the determinant of a matrix"
);
impl_expr_binary_func!(
    trace_of_product,
    dynetApplyTraceOfProduct,
    "Takes the trace of the product of matrices"
);

/// Performs layer normalization
pub fn layer_norm<E1: AsRef<Expression>, E2: AsRef<Expression>, E3: AsRef<Expression>>(
    x: E1,
    g: E2,
    b: E3,
) -> Expression {
    expr_func_body!(
        dynetApplyLayerNorm,
        x.as_ref().as_ptr(),
        g.as_ref().as_ptr(),
        b.as_ref().as_ptr()
    )
}

impl_expr_binary_func!(
    weight_norm,
    dynetApplyWeightNorm,
    "Performs weight normalization"
);

/// Copies tensor between devices
pub fn to_device<E: AsRef<Expression>>(x: E, device: &mut Device) -> Expression {
    expr_func_body!(dynetApplyToDevice, x.as_ref().as_ptr(), device.as_mut_ptr())
}
