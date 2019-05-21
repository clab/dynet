use std::ptr::{self, NonNull};

use dynet_sys;

use super::{
    ApiResult, ComputationGraph, Expression, Parameter, ParameterCollection, Result, Wrap,
};

/// `SoftmaxBuilder` trait
pub trait SoftmaxBuilder: Wrap<dynet_sys::dynetSoftmaxBuilder_t> {
    /// Initializes the parameters in the computation graph.
    fn new_graph(&mut self, cg: &mut ComputationGraph, update: bool) {
        unsafe {
            check_api_status!(dynet_sys::dynetResetSoftmaxBuilderGraph(
                self.as_mut_ptr(),
                cg.as_mut_ptr(),
                update as u32,
            ));
        }
    }

    /// Computes negative log probability of a class.
    fn neg_log_softmax_one<E: AsRef<Expression>>(&mut self, rep: E, classidx: u32) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetApplySoftmaxBuilderNegLogSoftmaxOne(
                self.as_mut_ptr(),
                rep.as_ref().as_ptr(),
                classidx,
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Computes batched negative log probability of a class.
    fn neg_log_softmax<E: AsRef<Expression>>(&mut self, rep: E, classidxs: &[u32]) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetApplySoftmaxBuilderNegLogSoftmax(
                self.as_mut_ptr(),
                rep.as_ref().as_ptr(),
                classidxs.as_ptr(),
                classidxs.len(),
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Samples from the softmax distribution.
    fn sample<E: AsRef<Expression>>(&mut self, rep: E) -> u32 {
        unsafe {
            let mut retval: u32 = 0;
            check_api_status!(dynet_sys::dynetSampleFromSoftmaxBuilder(
                self.as_mut_ptr(),
                rep.as_ref().as_ptr(),
                &mut retval,
            ));
            retval
        }
    }

    /// Returns an Expression representing a vector the size of the number of classes.
    fn full_log_distribution<E: AsRef<Expression>>(&mut self, rep: E) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetSoftmaxBuilderFullLogDistribution(
                self.as_mut_ptr(),
                rep.as_ref().as_ptr(),
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Returns the logits (before application of the softmax).
    fn full_logits<E: AsRef<Expression>>(&mut self, rep: E) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetSoftmaxBuilderFullLogits(
                self.as_mut_ptr(),
                rep.as_ref().as_ptr(),
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Gets the ParameterCollection containing the softmax parameters.
    fn get_parameter_collection(&mut self) -> ParameterCollection {
        unsafe {
            let mut pc_ptr: *mut dynet_sys::dynetParameterCollection_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetSoftmaxBuilderParameterCollection(
                self.as_mut_ptr(),
                &mut pc_ptr,
            ));
            ParameterCollection::from_raw(pc_ptr, false)
        }
    }
}

macro_rules! impl_softmax_builder {
    ($name:ident) => {
        impl_wrap_owned!($name, dynetSoftmaxBuilder_t);
        impl_drop!($name, dynetDeleteSoftmaxBuilder);
        impl SoftmaxBuilder for $name {}
    };
}

/// A builder for the standard softmax.
#[derive(Debug)]
pub struct StandardSoftmaxBuilder {
    inner: NonNull<dynet_sys::dynetSoftmaxBuilder_t>,
}

impl_softmax_builder!(StandardSoftmaxBuilder);

impl StandardSoftmaxBuilder {
    /// Creates a new `StandardSoftmaxBuilder`.
    ///
    /// # Arguments
    ///
    /// * rep_dim - Dimension of the input vectors.
    /// * num_classes - Number of classes.
    /// * pc - Parameter collection.
    /// * bias - Whether to use a bias vector or not.
    pub fn new(
        rep_dim: u32,
        num_classes: u32,
        pc: &mut ParameterCollection,
        bias: bool,
    ) -> StandardSoftmaxBuilder {
        unsafe {
            let mut builder_ptr: *mut dynet_sys::dynetSoftmaxBuilder_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateStandardSoftmaxBuilder(
                rep_dim,
                num_classes,
                pc.as_mut_ptr(),
                bias as u32,
                &mut builder_ptr,
            ));
            StandardSoftmaxBuilder::from_raw(builder_ptr, true)
        }
    }

    /// Creates a new `StandardSoftmaxBuilder` from parameters.
    pub fn from_parameters(p_w: &mut Parameter, p_b: &mut Parameter) -> StandardSoftmaxBuilder {
        unsafe {
            let mut builder_ptr: *mut dynet_sys::dynetSoftmaxBuilder_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateStandardSoftmaxBuilderFromParameters(
                p_w.as_mut_ptr(),
                p_b.as_mut_ptr(),
                &mut builder_ptr,
            ));
            StandardSoftmaxBuilder::from_raw(builder_ptr, true)
        }
    }
}
