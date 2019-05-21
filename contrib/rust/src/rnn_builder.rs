use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, ComputationGraph, Expression, ParameterCollection, Result, Wrap};

/// `RNNBuilder` trait
pub trait RNNBuilder: Wrap<dynet_sys::dynetRNNBuilder_t> {
    /// Gets pointer to the current state.
    fn state(&self) -> i32 {
        unsafe {
            let mut retval: i32 = 0;
            check_api_status!(dynet_sys::dynetGetRNNBuilderStatePointer(
                self.as_ptr(),
                &mut retval,
            ));
            retval
        }
    }

    /// Resets the internally used computation graph with a new one.
    fn new_graph(&mut self, cg: &mut ComputationGraph, update: bool) {
        unsafe {
            check_api_status!(dynet_sys::dynetResetRNNBuilderGraph(
                self.as_mut_ptr(),
                cg.as_mut_ptr(),
                update as u32,
            ));
        }
    }

    /// Resets the builder for a new sequence.
    fn start_new_sequence<ES: AsRef<[E]>, E: AsRef<Expression>>(&mut self, h_0: ES) {
        unsafe {
            let h_ptrs: Vec<_> = h_0.as_ref().iter().map(|h| h.as_ref().as_ptr()).collect();
            check_api_status!(dynet_sys::dynetStartRNNBuilderNewSequence(
                self.as_mut_ptr(),
                h_ptrs.as_ptr(),
                h_ptrs.len(),
            ));
        }
    }

    /// Sets the output state of a node.
    fn set_h<ES: AsRef<[E]>, E: AsRef<Expression>>(&mut self, prev: i32, h_new: ES) {
        unsafe {
            let h_ptrs: Vec<_> = h_new.as_ref().iter().map(|h| h.as_ref().as_ptr()).collect();
            check_api_status!(dynet_sys::dynetSetRNNBuilderHiddenState(
                self.as_mut_ptr(),
                prev,
                h_ptrs.as_ptr(),
                h_ptrs.len(),
            ));
        }
    }

    /// Sets the internal state of a node.
    fn set_s<ES: AsRef<[E]>, E: AsRef<Expression>>(&mut self, prev: i32, c_new: ES) {
        unsafe {
            let c_ptrs: Vec<_> = c_new.as_ref().iter().map(|c| c.as_ref().as_ptr()).collect();
            check_api_status!(dynet_sys::dynetSetRNNBuilderCellState(
                self.as_mut_ptr(),
                prev,
                c_ptrs.as_ptr(),
                c_ptrs.len(),
            ));
        }
    }

    /// Adds another timestep by reading in the variable x.
    fn add_input<E: AsRef<Expression>>(&mut self, x: E) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetAddRNNBuilderInput(
                self.as_mut_ptr(),
                x.as_ref().as_ptr(),
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Adds another timestep, with arbitrary recurrent connection.
    fn add_input_to_state<E: AsRef<Expression>>(&mut self, prev: i32, x: E) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetAddRNNBuilderInputToState(
                self.as_mut_ptr(),
                prev,
                x.as_ref().as_ptr(),
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Rewinds the last timestep.
    fn rewind_one_step(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetRewindRNNBuilderOneStep(self.as_mut_ptr()));
        }
    }

    /// Returns the RNN state that is the parent of the given state.
    fn get_head(&self, p: i32) -> i32 {
        unsafe {
            let mut retval: i32 = 0;
            check_api_status!(dynet_sys::dynetGetRNNBuilderParentStatePointer(
                self.as_ptr(),
                p,
                &mut retval,
            ));
            retval
        }
    }

    /// Sets dropout.
    fn set_dropout(&mut self, d: f32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetRNNBuilderDropout(self.as_mut_ptr(), d));
        }
    }

    /// Disables dropout.
    fn disable_dropout(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetDisableRNNBuilderDropout(self.as_mut_ptr()));
        }
    }

    /// Returns node (index) of most recent output.
    fn back(&self) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetRNNBuilderLastOutput(
                self.as_ptr(),
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Returns the final output of each hidden layer.
    fn final_h(&self) -> Vec<Expression> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(dynet_sys::dynetGetRNNBuilderFinalHiddenState(
                self.as_ptr(),
                ptr::null_mut(),
                &mut size,
            ));
            let mut expr_ptrs = vec![ptr::null_mut(); size];
            check_api_status!(dynet_sys::dynetGetRNNBuilderFinalHiddenState(
                self.as_ptr(),
                expr_ptrs.as_mut_ptr(),
                &mut size,
            ));
            expr_ptrs
                .into_iter()
                .map(|expr_ptr| Expression::from_raw(expr_ptr, true))
                .collect()
        }
    }

    /// Returns the output of any hidden layer.
    fn get_h(&self, i: i32) -> Vec<Expression> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(dynet_sys::dynetGetRNNBuilderHiddenState(
                self.as_ptr(),
                i,
                ptr::null_mut(),
                &mut size,
            ));
            let mut expr_ptrs = vec![ptr::null_mut(); size];
            check_api_status!(dynet_sys::dynetGetRNNBuilderHiddenState(
                self.as_ptr(),
                i,
                expr_ptrs.as_mut_ptr(),
                &mut size,
            ));
            expr_ptrs
                .into_iter()
                .map(|expr_ptr| Expression::from_raw(expr_ptr, true))
                .collect()
        }
    }

    /// Returns the final state of each hidden layer.
    fn final_s(&self) -> Vec<Expression> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(dynet_sys::dynetGetRNNBuilderFinalCellState(
                self.as_ptr(),
                ptr::null_mut(),
                &mut size,
            ));
            let mut expr_ptrs = vec![ptr::null_mut(); size];
            check_api_status!(dynet_sys::dynetGetRNNBuilderFinalCellState(
                self.as_ptr(),
                expr_ptrs.as_mut_ptr(),
                &mut size,
            ));
            expr_ptrs
                .into_iter()
                .map(|expr_ptr| Expression::from_raw(expr_ptr, true))
                .collect()
        }
    }

    /// Returns the state of any hidden layer.
    fn get_s(&self, i: i32) -> Vec<Expression> {
        unsafe {
            let mut size: usize = 0;
            check_api_status!(dynet_sys::dynetGetRNNBuilderCellState(
                self.as_ptr(),
                i,
                ptr::null_mut(),
                &mut size,
            ));
            let mut expr_ptrs = vec![ptr::null_mut(); size];
            check_api_status!(dynet_sys::dynetGetRNNBuilderCellState(
                self.as_ptr(),
                i,
                expr_ptrs.as_mut_ptr(),
                &mut size,
            ));
            expr_ptrs
                .into_iter()
                .map(|expr_ptr| Expression::from_raw(expr_ptr, true))
                .collect()
        }
    }

    /// Returns the number of components in `h_0`.
    fn num_h0_components(&self) -> i32 {
        unsafe {
            let mut retval: i32 = 0;
            check_api_status!(dynet_sys::dynetGetRNNBuilderNumH0Components(
                self.as_ptr(),
                &mut retval,
            ));
            retval
        }
    }

    /// Copies the parameters of another builder.
    fn copy<B: RNNBuilder>(&mut self, params: &B) {
        unsafe {
            check_api_status!(dynet_sys::dynetCopyRNNBuilderParameters(
                self.as_mut_ptr(),
                params.as_ptr(),
            ));
        }
    }

    /// Gets parameters in the RNNBuilder.
    fn get_parameter_collection(&mut self) -> ParameterCollection {
        unsafe {
            let mut pc_ptr: *mut dynet_sys::dynetParameterCollection_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetGetRNNBuilderParameterCollection(
                self.as_mut_ptr(),
                &mut pc_ptr,
            ));
            ParameterCollection::from_raw(pc_ptr, false)
        }
    }
}

macro_rules! impl_rnn_builder {
    ($name:ident) => {
        impl_wrap_owned!($name, dynetRNNBuilder_t);
        impl_drop!($name, dynetDeleteRNNBuilder);
        impl RNNBuilder for $name {}
    };
}

/// A builder for the simplest RNN with tanh nonlinearity.
#[derive(Debug)]
pub struct SimpleRNNBuilder {
    inner: NonNull<dynet_sys::dynetRNNBuilder_t>,
}

impl_rnn_builder!(SimpleRNNBuilder);

impl SimpleRNNBuilder {
    /// Creates a new `SimpleRNNBuilder`.
    ///
    /// # Arguments
    ///
    /// * layers - Number of layers.
    /// * input_dim - Dimension of the input.
    /// * hidden_dim - Hidden layer (and output) size.
    /// * model - ParameterCollection holding the parameters.
    /// * support_lags - Allow for auxiliary output.
    pub fn new(
        layers: u32,
        input_dim: u32,
        hidden_dim: u32,
        model: &mut ParameterCollection,
        support_lags: bool,
    ) -> SimpleRNNBuilder {
        unsafe {
            let mut builder_ptr: *mut dynet_sys::dynetRNNBuilder_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateSimpleRNNBuilder(
                layers,
                input_dim,
                hidden_dim,
                model.as_mut_ptr(),
                support_lags as u32,
                &mut builder_ptr,
            ));
            SimpleRNNBuilder::from_raw(builder_ptr, true)
        }
    }

    /// Adds auxiliary output.
    pub fn add_auxiliary_input<E1: AsRef<Expression>, E2: AsRef<Expression>>(
        &mut self,
        x: E1,
        aux: E2,
    ) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetAddSimpleRNNBuilderAuxiliaryInput(
                self.as_mut_ptr(),
                x.as_ref().as_ptr(),
                aux.as_ref().as_ptr(),
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Sets the dropout rates.
    pub fn set_variational_dropout(&mut self, d: f32, d_h: f32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetSimpleRNNBuilderDropout(
                self.as_mut_ptr(),
                d,
                d_h,
            ));
        }
    }

    /// Sets dropout masks at the beginning of a sequence for a specific batch size.
    pub fn set_dropout_masks(&mut self, batch_size: u32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetSimpleRNNBuilderDropoutMasks(
                self.as_mut_ptr(),
                batch_size,
            ));
        }
    }
}

/// A builder for an LSTM unit with coupled input and forget gate as well as peepholes connections.
#[derive(Debug)]
pub struct CoupledLSTMBuilder {
    inner: NonNull<dynet_sys::dynetRNNBuilder_t>,
}

impl_rnn_builder!(CoupledLSTMBuilder);

impl CoupledLSTMBuilder {
    /// Creates a new `CoupledLSTMBuilder`.
    ///
    /// # Arguments
    ///
    /// * layers - Number of layers.
    /// * input_dim - Dimention of the input \f$x_t\f$.
    /// * hidden_dim - Dimention of the hidden states \f$h_t\f$ and \f$c_t\f$.
    /// * model ParameterCollection holding the parameters.
    pub fn new(
        layers: u32,
        input_dim: u32,
        hidden_dim: u32,
        model: &mut ParameterCollection,
    ) -> CoupledLSTMBuilder {
        unsafe {
            let mut builder_ptr: *mut dynet_sys::dynetRNNBuilder_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateCoupledLSTMBuilder(
                layers,
                input_dim,
                hidden_dim,
                model.as_mut_ptr(),
                &mut builder_ptr,
            ));
            CoupledLSTMBuilder::from_raw(builder_ptr, true)
        }
    }

    /// Sets the dropout rates.
    pub fn set_variational_dropout(&mut self, d: f32, d_h: f32, d_c: f32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetCoupledLSTMBuilderDropout(
                self.as_mut_ptr(),
                d,
                d_h,
                d_c,
            ));
        }
    }

    /// Sets dropout masks at the beginning of a sequence for a specific batch size.
    pub fn set_dropout_masks(&mut self, batch_size: u32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetCoupledLSTMBuilderDropoutMasks(
                self.as_mut_ptr(),
                batch_size,
            ));
        }
    }
}

/// A builder for a "standard" LSTM, ie with decoupled input and forget gates and no peephole
/// connections.
#[derive(Debug)]
pub struct VanillaLSTMBuilder {
    inner: NonNull<dynet_sys::dynetRNNBuilder_t>,
}

impl_rnn_builder!(VanillaLSTMBuilder);

impl VanillaLSTMBuilder {
    /// Creates a new `VanillaLSTMBuilder`.
    ///
    /// # Arguments
    ///
    /// * layers - Number of layers.
    /// * input_dim - Dimention of the input \f$x_t\f$.
    /// * hidden_dim - Dimention of the hidden states \f$h_t\f$ and \f$c_t\f$.
    /// * model - ParameterCollection holding the parameters.
    /// * ln_lstm - Whether to use layer normalization.
    /// * forget_bias - A float value to use as bias for the forget gate.
    pub fn new(
        layers: u32,
        input_dim: u32,
        hidden_dim: u32,
        model: &mut ParameterCollection,
        ln_lstm: bool,
        forget_bias: f32,
    ) -> VanillaLSTMBuilder {
        unsafe {
            let mut builder_ptr: *mut dynet_sys::dynetRNNBuilder_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateVanillaLSTMBuilder(
                layers,
                input_dim,
                hidden_dim,
                model.as_mut_ptr(),
                ln_lstm as u32,
                forget_bias,
                &mut builder_ptr,
            ));
            VanillaLSTMBuilder::from_raw(builder_ptr, true)
        }
    }

    /// Sets the dropout rates.
    pub fn set_variational_dropout(&mut self, d: f32, d_r: f32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetVanillaLSTMBuilderDropout(
                self.as_mut_ptr(),
                d,
                d_r,
            ));
        }
    }

    /// Sets dropout masks at the beginning of a sequence for a specific batch size.
    pub fn set_dropout_masks(&mut self, batch_size: u32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetVanillaLSTMBuilderDropoutMasks(
                self.as_mut_ptr(),
                batch_size,
            ));
        }
    }
}

/// A builder for a "standard" LSTM, ie with decoupled input and forget gates and no peephole
/// connections.
#[derive(Debug)]
pub struct CompactVanillaLSTMBuilder {
    inner: NonNull<dynet_sys::dynetRNNBuilder_t>,
}

impl_rnn_builder!(CompactVanillaLSTMBuilder);

impl CompactVanillaLSTMBuilder {
    /// Creates a new `CompactVanillaLSTMBuilder`.
    ///
    /// # Arguments
    ///
    /// * layers - Number of layers.
    /// * input_dim - Dimention of the input \f$x_t\f$.
    /// * hidden_dim - Dimention of the hidden states \f$h_t\f$ and \f$c_t\f$.
    /// * model - ParameterCollection holding the parameters.
    pub fn new(
        layers: u32,
        input_dim: u32,
        hidden_dim: u32,
        model: &mut ParameterCollection,
    ) -> CompactVanillaLSTMBuilder {
        unsafe {
            let mut builder_ptr: *mut dynet_sys::dynetRNNBuilder_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateCompactVanillaLSTMBuilder(
                layers,
                input_dim,
                hidden_dim,
                model.as_mut_ptr(),
                &mut builder_ptr,
            ));
            CompactVanillaLSTMBuilder::from_raw(builder_ptr, true)
        }
    }

    /// Sets the dropout rates.
    pub fn set_variational_dropout(&mut self, d: f32, d_r: f32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetCompactVanillaLSTMBuilderDropout(
                self.as_mut_ptr(),
                d,
                d_r,
            ));
        }
    }

    /// Sets dropout masks at the beginning of a sequence for a specific batch size.
    pub fn set_dropout_masks(&mut self, batch_size: u32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetCompactVanillaLSTMBuilderDropoutMasks(
                self.as_mut_ptr(),
                batch_size,
            ));
        }
    }

    /// Sets the gaussian weight noise.
    pub fn set_weightnoise(&mut self, std: f32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetCompactVanillaLSTMBuilderWeightnoise(
                self.as_mut_ptr(),
                std,
            ));
        }
    }
}

/// FastLSTMBuilder
#[derive(Debug)]
pub struct FastLSTMBuilder {
    inner: NonNull<dynet_sys::dynetRNNBuilder_t>,
}

impl_rnn_builder!(FastLSTMBuilder);

impl FastLSTMBuilder {
    /// Creates a new `FastLSTMBuilder`.
    ///
    /// # Arguments
    ///
    /// * layers - Number of layers.
    /// * input_dim - Dimention of the input \f$x_t\f$.
    /// * hidden_dim - Dimention of the hidden states \f$h_t\f$ and \f$c_t\f$.
    /// * model - ParameterCollection holding the parameters.
    pub fn new(
        layers: u32,
        input_dim: u32,
        hidden_dim: u32,
        model: &mut ParameterCollection,
    ) -> FastLSTMBuilder {
        unsafe {
            let mut builder_ptr: *mut dynet_sys::dynetRNNBuilder_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateFastLSTMBuilder(
                layers,
                input_dim,
                hidden_dim,
                model.as_mut_ptr(),
                &mut builder_ptr,
            ));
            FastLSTMBuilder::from_raw(builder_ptr, true)
        }
    }
}

/// `TreeLSTMBuilder` trait
pub trait TreeLSTMBuilder: RNNBuilder {
    /// Adds input with given children at position id.
    fn add_input_to_children<E: AsRef<Expression>>(
        &mut self,
        id: i32,
        children: &mut [i32],
        x: E,
    ) -> Expression {
        unsafe {
            let mut expr_ptr: *mut dynet_sys::dynetExpression_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetAddTreeLSTMBuilderInput(
                self.as_mut_ptr(),
                id,
                children.as_mut_ptr(),
                children.len(),
                x.as_ref().as_ptr(),
                &mut expr_ptr,
            ));
            Expression::from_raw(expr_ptr, true)
        }
    }

    /// Sets the number of nodes in the tree in advance.
    fn set_num_elements(&mut self, num: i32) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetTreeLSTMBuilderNumElements(
                self.as_mut_ptr(),
                num,
            ));
        }
    }
}

macro_rules! impl_tree_lstm_builder {
    ($name:ident, $create_fn:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $name {
            inner: NonNull<dynet_sys::dynetRNNBuilder_t>,
        }

        impl_tree_lstm_builder!($name);

        impl $name {
            /// Creates a new `$name`.
            ///
            /// # Arguments
            ///
            /// * layers - Number of layers.
            /// * input_dim - Dimention of the input.
            /// * hidden_dim - Dimention of the hidden states.
            /// * model - ParameterCollection holding the parameters.
            pub fn new(
                layers: u32,
                input_dim: u32,
                hidden_dim: u32,
                model: &mut ParameterCollection,
            ) -> $name {
                unsafe {
                    let mut builder_ptr: *mut dynet_sys::dynetRNNBuilder_t = ptr::null_mut();
                    check_api_status!(dynet_sys::$create_fn(
                        layers,
                        input_dim,
                        hidden_dim,
                        model.as_mut_ptr(),
                        &mut builder_ptr,
                    ));
                    $name::from_raw(builder_ptr, true)
                }
            }
        }
    };
    ($name:ident) => {
        impl_rnn_builder!($name);
        impl TreeLSTMBuilder for $name {}
    };
}

/// A builder for N-ary trees with a fixed upper bound of children.
#[derive(Debug)]
pub struct NaryTreeLSTMBuilder {
    inner: NonNull<dynet_sys::dynetRNNBuilder_t>,
}

impl_tree_lstm_builder!(NaryTreeLSTMBuilder);

impl NaryTreeLSTMBuilder {
    /// Creates a new `NaryTreeLSTMBuilder`.
    ///
    /// # Arguments
    ///
    /// * n - Max branching factor.
    /// * layers - Number of layers.
    /// * input_dim - Dimention of the input \f$x_t\f$.
    /// * hidden_dim - Dimention of the hidden states \f$h_t\f$ and \f$c_t\f$.
    /// * model - ParameterCollection holding the parameters.
    pub fn new(
        n: u32,
        layers: u32,
        input_dim: u32,
        hidden_dim: u32,
        model: &mut ParameterCollection,
    ) -> NaryTreeLSTMBuilder {
        unsafe {
            let mut builder_ptr: *mut dynet_sys::dynetRNNBuilder_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateNaryTreeLSTMBuilder(
                n,
                layers,
                input_dim,
                hidden_dim,
                model.as_mut_ptr(),
                &mut builder_ptr,
            ));
            NaryTreeLSTMBuilder::from_raw(builder_ptr, true)
        }
    }
}

impl_tree_lstm_builder!(
    UnidirectionalTreeLSTMBuilder,
    dynetCreateUnidirectionalTreeLSTMBuilder,
    "A builder for a tree-LSTM which is recursively defined by a unidirectional LSTM over the node
    and its children representations."
);

impl_tree_lstm_builder!(
    BidirectionalTreeLSTMBuilder,
    dynetCreateBidirectionalTreeLSTMBuilder,
    "A builder for a tree-LSTM which is recursively defined by a bidirectional LSTM over the node
    and its children representations."
);

/// GRUBuilder
#[derive(Debug)]
pub struct GRUBuilder {
    inner: NonNull<dynet_sys::dynetRNNBuilder_t>,
}

impl_rnn_builder!(GRUBuilder);

impl GRUBuilder {
    /// Creates a new `GRUBuilder`.
    ///
    /// # Arguments
    ///
    /// * layers - Number of layers.
    /// * input_dim - Dimention of the input.
    /// * hidden_dim - Dimention of the hidden states.
    /// * model - ParameterCollection holding the parameters.
    pub fn new(
        layers: u32,
        input_dim: u32,
        hidden_dim: u32,
        model: &mut ParameterCollection,
    ) -> GRUBuilder {
        unsafe {
            let mut builder_ptr: *mut dynet_sys::dynetRNNBuilder_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateGRUBuilder(
                layers,
                input_dim,
                hidden_dim,
                model.as_mut_ptr(),
                &mut builder_ptr,
            ));
            GRUBuilder::from_raw(builder_ptr, true)
        }
    }
}
