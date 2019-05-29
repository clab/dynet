//! The `dynet` crate provides Rust interfaces for DyNet.
//!
//! DyNet is a neural network library developed by Carnegie Mellon University and many others. It
//! is written in C++ and is designed to be efficient when run on either CPU or GPU, and to work
//! well with networks that have dynamic structures that change for every training instance. For
//! example, these kinds of networks are particularly important in natural language processing
//! tasks, and DyNet has been used to build state-of-the-art systems for syntactic parsing, machine
//! translation, morphological inflection, and many other application areas.

#![deny(missing_docs)]
#![warn(
    missing_copy_implementations, missing_debug_implementations, trivial_casts,
    trivial_numeric_casts, unused_extern_crates, unused_import_braces, unused_qualifications
)]

extern crate backtrace;
extern crate dynet_sys;
extern crate libc;

#[macro_use]
mod status;
pub(crate) use status::*;

#[macro_use]
mod util;
pub use util::*;

mod device;
pub use device::Device;
mod dim;
pub use dim::Dim;
mod expr;
pub use expr::*;
mod graph;
pub use graph::*;
mod init;
pub use init::{initialize, reset_rng, DynetParams};
mod io;
pub use io::*;
mod model;
pub use model::{LookupParameter, Parameter, ParameterCollection};
mod param_init;
pub use param_init::*;
mod rnn_builder;
pub use rnn_builder::*;
mod softmax_builder;
pub use softmax_builder::*;
mod tensor;
pub use tensor::Tensor;
mod training;
pub use training::*;
