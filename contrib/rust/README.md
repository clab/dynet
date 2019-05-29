# Rust bindings for DyNet

This package provides the Rust bindings for [DyNet](https://github.com/clab/dynet).

## Building the Rust bindings

To build the Rust bindings, just run `cargo build` command:

```
$ cargo build [--features cuda]
```

You can specify the library directory using `DYNET_C_LIBRARY_DIR` containing `libdynet_c.so`.

## Running the Examples

You can run examples with `cargo run` command:

```
$ cargo run --example xor
```

## Usage

The current Rust API works mostly like the C++ API.

## Acknowledgements

The DyNet Rust bindings are designed in reference to [primitiv](https://github.com/primitiv/primitiv-rust).
