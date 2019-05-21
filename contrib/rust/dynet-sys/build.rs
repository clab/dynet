extern crate bindgen;

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{exit, Command};
use std::result::Result;

const FRAMEWORK_LIBRARY: &'static str = "dynet";
const LIBRARY: &'static str = "dynet_c";

macro_rules! log {
    ($fmt:expr) => (println!(concat!("dynet-sys/build.rs:{}: ", $fmt), line!()));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("dynet-sys/build.rs:{}: ", $fmt), line!(), $($arg)*));
}
macro_rules! log_var(($var:ident) => (log!(concat!(stringify!($var), " = {:?}"), $var)));

fn main() {
    let lib_dir = env::var("DYNET_C_LIBRARY_DIR").unwrap_or("/usr/local/lib".to_string());
    let library = format!("lib{}.so", LIBRARY);
    let library_path = Path::new(&lib_dir).join(&library);

    if !library_path.exists() {
        let no_install = match env::var("DYNET_NO_INSTALL") {
            Ok(s) => s != "0",
            Err(_) => false,
        };
        let force_build = match env::var("DYNET_FORCE_BUILD") {
            Ok(s) => s != "0",
            Err(_) => false,
        };

        let (output_lib_dir, output_include_dir) = match if force_build || true {
            // NOTE: currently only support building from source.
            build_from_src()
        } else {
            download_prebuild()
        } {
            Ok(dirs) => {
                log!("Successfully created `{}`.", library);
                dirs
            }
            Err(e) => panic!("Failed to create `{}`.\nreason: {}", library, e),
        };

        if !no_install {
            let include_dir =
                env::var("DYNET_C_INCLUDE_DIR").unwrap_or("/usr/local/include".to_string());
            let framework_library = format!("lib{}.so", FRAMEWORK_LIBRARY);
            let framework_library_path = Path::new(&lib_dir).join(&framework_library);
            let include_path = Path::new(&include_dir).join(LIBRARY);
            let framework_include_path = Path::new(&include_dir).join(FRAMEWORK_LIBRARY);
            if framework_include_path.exists() {
                run("rm", |command| {
                    command.arg("-rf").arg(&framework_include_path)
                });
            }
            run("cp", |command| {
                command
                    .arg("-r")
                    .arg(output_include_dir.join(FRAMEWORK_LIBRARY))
                    .arg(&framework_include_path)
            });
            if include_path.exists() {
                run("rm", |command| command.arg("-rf").arg(&include_path));
            }
            run("cp", |command| {
                command
                    .arg("-r")
                    .arg(output_include_dir.join(LIBRARY))
                    .arg(&include_path)
            });
            fs::copy(
                output_lib_dir.join(framework_library),
                framework_library_path,
            ).unwrap();
            fs::copy(output_lib_dir.join(library), library_path).unwrap();
        }
    }

    exit(match build_bindings() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}

fn download_prebuild() -> Result<((PathBuf, PathBuf)), Box<Error>> {
    Err("Not supported.".into())
}

fn build_from_src() -> Result<((PathBuf, PathBuf)), Box<Error>> {
    let tag = {
        let output = Command::new("git")
            .arg("rev-parse")
            .arg("--verify")
            .arg("HEAD")
            .output()?;
        assert!(output.status.success());
        String::from_utf8(output.stdout)?.trim().to_string()
    };
    let out_dir = PathBuf::from(env::var("OUT_DIR")?).join(format!("libdynet-{}", tag));
    log_var!(out_dir);
    if !out_dir.exists() {
        fs::create_dir(&out_dir)?;
    }
    let framework_library_path = out_dir.join(format!("lib/lib{}.so", FRAMEWORK_LIBRARY));
    log_var!(framework_library_path);
    let library_path = out_dir.join(format!("lib/lib{}.so", LIBRARY));
    log_var!(library_path);
    if library_path.exists() && framework_library_path.exists() {
        log!(
            "{:?} and {:?} already exist, not building",
            library_path,
            framework_library_path
        );
    } else {
        let build_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?)
            .join(format!("target/build-dynet-{}", tag));
        log_var!(build_dir);
        if !build_dir.exists() {
            fs::create_dir_all(&build_dir)?;
        }
        let build_dir_s = build_dir.to_str().unwrap();
        let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?)
            .ancestors()
            .skip(3)
            .next()
            .unwrap()
            .to_path_buf();
        run("cmake", |command| {
            command
                .arg(&source)
                .arg(format!("-B{}", build_dir_s))
                .arg(format!(
                    "-DCMAKE_INSTALL_PREFIX={}",
                    out_dir.to_str().unwrap()
                ))
                .arg(format!(
                    "-DEIGEN3_INCLUDE_DIR={}",
                    env::var("EIGEN3_INCLUDE_DIR").unwrap_or("/usr/local/lib".to_string())
                ))
                .arg("-DENABLE_C=ON")
                .arg(format!(
                    "-DBACKEND={}",
                    if cfg!(feature = "cuda") {
                        "cuda"
                    } else {
                        "eigen"
                    }
                ))
        });
        run("make", |command| {
            command
                .arg(format!("--directory={}", build_dir_s))
                .arg("-j4")
        });
        run("make", |command| {
            command
                .arg("install")
                .arg(format!("--directory={}", build_dir_s))
        });
    }
    assert!(library_path.exists());
    Ok((out_dir.join("lib"), out_dir.join("include")))
}

fn run<F>(name: &str, mut configure: F)
where
    F: FnMut(&mut Command) -> &mut Command,
{
    let mut command = Command::new(name);
    let configured = configure(&mut command);
    log!("Executing {:?}", configured);
    if !configured.status().unwrap().success() {
        panic!("failed to execute {:?}", configured);
    }
    log!("Command {:?} finished successfully", configured);
}

fn build_bindings() -> Result<(), Box<Error>> {
    let lib_dir = env::var("DYNET_C_LIBRARY_DIR").unwrap_or("/usr/local/lib".to_string());
    let include_dir = env::var("DYNET_C_INCLUDE_DIR").unwrap_or("/usr/local/include".to_string());
    println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
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
