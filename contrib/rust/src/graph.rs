use std::ffi::CString;
use std::io as std_io;
use std::path::Path;
use std::ptr::{self, NonNull};

use dynet_sys;

use super::{ApiResult, Expression, Result, Tensor, Wrap};

/// Computation graph where nodes represent forward and backward intermediate values, and edges
/// represent functions of multiple values.
// TODO(chantera): write example.
#[derive(Debug)]
pub struct ComputationGraph {
    inner: NonNull<dynet_sys::dynetComputationGraph_t>,
}

impl_wrap_owned!(ComputationGraph, dynetComputationGraph_t);
impl_drop!(ComputationGraph, dynetDeleteComputationGraph);

impl ComputationGraph {
    /// Creates a new `ComputationGraph`.
    pub fn new() -> ComputationGraph {
        unsafe {
            let mut graph_ptr: *mut dynet_sys::dynetComputationGraph_t = ptr::null_mut();
            check_api_status!(dynet_sys::dynetCreateComputationGraph(&mut graph_ptr));
            ComputationGraph::from_raw(graph_ptr, true)
        }
    }

    /// Reset the graph to a newly created state.
    pub fn clear(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetClearComputationGraph(self.as_mut_ptr()));
        }
    }

    /// Set a checkpoint.
    pub fn set_checkpoint(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetSetComputationGraphCheckpoint(
                self.as_mut_ptr()
            ));
        }
    }

    /// Revert the graph to the last checkpoint.
    pub fn revert(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetRevertComputationGraph(self.as_mut_ptr()));
        }
    }

    /// Runs complete forward pass from first node to given one, ignoring all precomputed values.
    pub fn forward(&mut self, last: &Expression) -> Tensor {
        unsafe {
            let mut tensor_ptr: *const dynet_sys::dynetTensor_t = ptr::null();
            check_api_status!(dynet_sys::dynetForwardExprOnComputationGraph(
                self.as_mut_ptr(),
                last.as_ptr(),
                &mut tensor_ptr,
            ));
            Tensor::from_raw(tensor_ptr as *mut _, false)
        }
    }

    /// Runs forward pass from first node to given one.
    pub fn incremental_forward(&mut self, last: &Expression) -> Tensor {
        unsafe {
            let mut tensor_ptr: *const dynet_sys::dynetTensor_t = ptr::null();
            check_api_status!(dynet_sys::dynetForwardExprIncrementallyOnComputationGraph(
                self.as_mut_ptr(),
                last.as_ptr(),
                &mut tensor_ptr,
            ));
            Tensor::from_raw(tensor_ptr as *mut _, false)
        }
    }

    /// Gets the gradient for the given expression.
    pub fn get_gradient(&mut self, last: &Expression) -> Tensor {
        unsafe {
            let mut tensor_ptr: *const dynet_sys::dynetTensor_t = ptr::null();
            check_api_status!(dynet_sys::dynetGetExprGradientFromComputationGraph(
                self.as_mut_ptr(),
                last.as_ptr(),
                &mut tensor_ptr,
            ));
            Tensor::from_raw(tensor_ptr as *mut _, false)
        }
    }

    /// Clears caches.
    pub fn invalidate(&mut self) {
        unsafe {
            check_api_status!(dynet_sys::dynetInvalidateComputationGraph(
                self.as_mut_ptr()
            ));
        }
    }

    /// Computes backward gradients from the front-most evaluated node.
    pub fn backward(&mut self, last: &Expression) {
        unsafe {
            check_api_status!(dynet_sys::dynetBackwardExprOnComputationGraph(
                self.as_mut_ptr(),
                last.as_ptr(),
            ));
        }
    }

    /// Visualizes the ComputationGraph.
    pub fn print_graphviz(&self) {
        unsafe {
            check_api_status!(dynet_sys::dynetPrintComputationGraphViz(self.as_ptr()));
        }
    }

    /// Dump the ComputationGraph.
    pub fn dump<P: AsRef<Path>>(
        &mut self,
        path: P,
        show_values: bool,
        show_gradients: bool,
        nan_check_only: bool,
    ) -> std_io::Result<()> {
        unsafe {
            let path_c = CString::new(path.as_ref().to_str().unwrap()).unwrap();
            Result::from_api_status(
                dynet_sys::dynetDumpComputationGraph(
                    self.as_mut_ptr(),
                    path_c.as_ptr(),
                    show_values as u32,
                    show_gradients as u32,
                    nan_check_only as u32,
                ),
                (),
            ).map_err(|status| std_io::Error::new(std_io::ErrorKind::Other, status.message()))
        }
    }
}

impl Default for ComputationGraph {
    fn default() -> ComputationGraph {
        ComputationGraph::new()
    }
}
