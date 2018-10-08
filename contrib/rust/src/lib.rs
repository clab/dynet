//! The `dynet` crate provides Rust interfaces for DyNet.
//!
//! DyNet is a neural network library developed by Carnegie Mellon University and many others. It
//! is written in C++ and is designed to be efficient when run on either CPU or GPU, and to work
//! well with networks that have dynamic structures that change for every training instance. For
//! example, these kinds of networks are particularly important in natural language processing
//! tasks, and DyNet has been used to build state-of-the-art systems for syntactic parsing, machine
//! translation, morphological inflection, and many other application areas.

#![deny(missing_docs)]

extern crate backtrace;
extern crate dynet_sys;
extern crate libc;

#[macro_use]
mod status;
pub(crate) use status::*;

#[macro_use]
mod util;
pub use util::*;

mod dim;
pub use dim::Dim;
