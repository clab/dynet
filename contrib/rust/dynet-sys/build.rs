extern crate bindgen;

use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process::exit;
use std::result::Result;

fn main() {
    exit(match build_bindings() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}

fn build_bindings() -> Result<(), Box<Error>> {
    let lib_dir = env::var("DYNET_C_LIBRARY_DIR").unwrap_or("/usr/local/lib".to_string());
    let include_dir = env::var("DYNET_C_INCLUDE_DIR").unwrap_or("/usr/local/include".to_string());
    println!("cargo:rustc-link-lib=dylib=dynet_c");
    println!("cargo:rustc-link-search={}", lib_dir);

    let builder = bindgen::Builder::default()
        .clang_arg(format!("-I{}", include_dir))
        .header(format!("{}/dynet_c/api.h", include_dir))
        .rustfmt_bindings(false)
        .generate_comments(false);

    builder
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(PathBuf::from(env::var("OUT_DIR")?).join("bindings.rs"))
        .expect("Couldn't write bindings!");
    Ok(())
}
