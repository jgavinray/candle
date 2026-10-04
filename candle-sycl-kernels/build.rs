// Compiles the SYCL C++ sources in `csrc/` into a shared library with `icpx`
// and links it into the crate. Runs ONLY when `candle-core` is built with
// `--features sycl` (this crate is excluded from the workspace), so a plain
// `cargo build` never reaches here and never needs oneAPI installed.
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCES: &[&str] = &[
    "runtime.cpp",
    "elementwise.cpp",
    "reduce.cpp",
    "ternary.cpp",
    "indexing.cpp",
    "pool.cpp",
    "im2col.cpp",
    "conv_transpose.cpp",
    "quant.cpp",
    "mmvq.cpp",
    "indexed_moe.cpp",
    "norm.cpp",
    "gemm.cpp",
];
// Every header the sources include, so that editing one rebuilds the library.
const HEADERS: &[&str] = &[
    "candle_sycl.h",
    "common.hpp",
    "quant_blocks.hpp",
    "mmvq_internal.hpp",
];

/// The icpx binary's resolved path; also watched, so an icpx upgrade
/// re-captures the version constant instead of keeping the stale one.
fn find_icpx() -> (PathBuf, String) {
    if let Ok(p) = std::env::var("CANDLE_SYCL_ICPX") {
        let p = PathBuf::from(&p);
        // canonicalize for a meaningful rerun-if-changed path; fall back to
        // the raw value if the fs call fails (cargo only warns on misses).
        let watched = p.canonicalize().unwrap_or_else(|_| p.clone());
        return (watched, p.display().to_string());
    }
    if Command::new("icpx").arg("--version").output().is_ok() {
        // Relative name: cargo resolves it against the current directory, so
        // there is no absolute path to watch. The version is still captured
        // on every fresh build-script run.
        return (PathBuf::from("icpx"), "icpx".to_string());
    }
    for root in [
        std::env::var("ONEAPI_ROOT").unwrap_or_default(),
        "/opt/intel/oneapi".to_string(),
    ] {
        if root.is_empty() {
            continue;
        }
        let cand = format!("{root}/compiler/latest/bin/icpx");
        if Path::new(&cand).exists() {
            // Both the watched path and the invoked name are the same
            // absolute candidate here.
            return (PathBuf::from(&cand), cand.clone());
        }
    }
    panic!(
        "candle-sycl-kernels: `icpx` not found. Install Intel oneAPI (or run in the \
         `intel/oneapi-basekit` container) and ensure `icpx` is on PATH, or set \
         CANDLE_SYCL_ICPX to its full path. This crate is only built for \
         `--features sycl`."
    );
}
/// The oneAPI release the `icpx --version` banner names — e.g. `2026.1.0`
/// from `Intel(R) oneAPI DPC++/C++ Compiler 2026.1.0 (2026.1.0.20260708)`.
/// Captured from the same binary that compiled the kernels, so the device
/// identity can name it at run time (tobert/lfm2d#3 item 8). Fails loudly
/// rather than baking a guessed default.
fn icpx_release(icpx: &str) -> Result<String, String> {
    let out = Command::new(icpx)
        .arg("--version")
        .output()
        .map_err(|e| format!("`{icpx} --version`: {e}"))?;
    if !out.status.success() {
        return Err(format!("`{icpx} --version` exited {}", out.status));
    }
    parse_icpx_release(&String::from_utf8_lossy(&out.stdout))
}

/// The first whitespace-delimited release token in the banner: two to four
/// dot-separated all-digit components whose first component is a four-digit
/// year (the whole oneAPI versioning scheme is `YYYY.N[.P]`, and the
/// parenthesised build id `2022.2.1.20221020` sits behind the bare release
/// token, so first-match is the right pick).
fn parse_icpx_release(text: &str) -> Result<String, String> {
    text.split_whitespace().find_map(|tok| {
        let parts: Vec<&str> = tok.split('.').collect();
        let shaped = parts.len() >= 2
            && parts.len() <= 4
            && parts[0].len() == 4
            && parts.iter().all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
        shaped.then(|| tok.to_string())
    }).ok_or_else(|| "no YYYY.N[.P] release token in `icpx --version` output".to_string())
}


fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc = manifest.join("csrc");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let (icpx_watch, icpx) = find_icpx();

    for f in SOURCES.iter().chain(HEADERS) {
        println!("cargo:rerun-if-changed=csrc/{f}");
    }
    println!("cargo:rerun-if-env-changed=CANDLE_SYCL_ICPX");
    // Watch the resolved binary itself: a oneAPI upgrade re-points
    // `compiler/latest` (and PATH), and the captured release must not go
    // stale. Only an absolute path is watched — cargo re-runs a build
    // script unconditionally when a rerun-if-changed path is missing, and
    // the relative PATH fallback is not one.
    if icpx_watch.is_absolute() {
        println!("cargo:rerun-if-changed={}", icpx_watch.display());
    }
    // Fail before the (minutes-long) kernel compiles if the banner cannot
    // be parsed: the identity constant must name the icpx that really
    // built this .so, never a guessed default (tobert/lfm2d#3 item 8).
    let release = icpx_release(&icpx).unwrap_or_else(|e| panic!("candle-sycl-kernels: {e}"));

    let common = [
        "-fsycl",
        "-O2",
        "-fPIC",
        "-std=c++20",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations",
    ];

    // The 13 translation units are independent; compile them concurrently.
    // A kernels rebuild is dominated by this stage, and the sequential loop
    // left 64 cores idle.
    let objs: Vec<PathBuf> = std::thread::scope(|s| -> Vec<PathBuf> {
        let handles: Vec<_> = SOURCES
            .iter()
            .map(|src| {
                let icpx = icpx.clone();
                let csrc = csrc.clone();
                let out = out.clone();
                s.spawn(move || {
                    let obj = out.join(format!("{src}.o"));
                    let status = Command::new(&icpx)
                        .args(common)
                        .arg("-qmkl=sequential")
                        .arg("-c")
                        .arg(csrc.join(src))
                        .arg("-o")
                        .arg(&obj)
                        .status()
                        .unwrap_or_else(|e| panic!("failed to spawn icpx for {src}: {e}"));
                    assert!(status.success(), "icpx failed to compile {src}");
                    obj
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("icpx worker panicked"))
            .collect()
    });

    // oneAPI runtime directories, baked into the .so as an rpath so that a
    // binary which finds this library resolves libsycl/libmkl without the
    // caller having sourced `setvars.sh`.
    let oneapi_root = std::env::var("ONEAPI_ROOT").ok().or_else(|| {
        Path::new("/opt/intel/oneapi")
            .exists()
            .then(|| "/opt/intel/oneapi".to_string())
    });
    let mut runtime_dirs: Vec<String> = Vec::new();
    if let Some(root) = oneapi_root.as_deref() {
        for sub in [
            "compiler/latest/lib",
            "mkl/latest/lib",
            "mkl/latest/lib/intel64",
            "tcm/latest/lib",
            "umf/latest/lib",
        ] {
            let p = format!("{root}/{sub}");
            if Path::new(&p).exists() {
                runtime_dirs.push(p);
            }
        }
    }

    let lib = out.join("libcandle_sycl.so");
    let status = Command::new(&icpx)
        .args(common)
        .arg("-shared")
        .args(&objs)
        .arg("-o")
        .arg(&lib)
        .args(runtime_dirs.iter().map(|d| format!("-L{d}")))
        .args(runtime_dirs.iter().map(|d| format!("-Wl,-rpath,{d}")))
        .arg("-qmkl=sequential")
        .arg("-lmkl_sycl")
        .arg("-lsycl")
        .status()
        .expect("failed to spawn icpx for link");
    assert!(status.success(), "icpx failed to link libcandle_sycl.so");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=candle_sycl");
    // So the .so is found at runtime without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out.display());
    // oneAPI runtime libs (libsycl, libmkl_*) live here.
    if let Ok(root) = std::env::var("ONEAPI_ROOT").or_else(|_| {
        if Path::new("/opt/intel/oneapi").exists() {
            Ok("/opt/intel/oneapi".to_string())
        } else {
            Err(std::env::VarError::NotPresent)
        }
    }) {
        for sub in [
            "compiler/latest/lib",
            "mkl/latest/lib",
            "mkl/latest/lib/intel64",
        ] {
            let p = format!("{root}/{sub}");
            if Path::new(&p).exists() {
                println!("cargo:rustc-link-search=native={p}");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
            }
        }
    }

    // Consumer metadata, namespaced by the Cargo.toml `links` key into
    // `DEP_CANDLE_SYCL_KERNELS_*` for the crate's direct dependents — the
    // successor to a downstream build script globbing the target dir for a
    // `candle-sycl-kernels-<hash>` directory and taking the newest-named hit,
    // which survived a candle bump by accident (tobert/lfm2d#3 item 8).
    // `icpx_version` is what the device identity names; `lib_dir` is where the
    // linked `libcandle_sycl.so` lives, so a final-link consumer (lfm2d's
    // build.rs) can rpath it without guessing.
    println!("cargo::metadata=icpx_version={release}");
    println!("cargo::metadata=lib_dir={}", out.display());
    // The same release, as a compile-time constant for this crate
    // (re-exported by candle-core) — replacing the run-time
    // `CANDLE_SYCL_ICPX_VERSION` env read whose silent 2026.1 default could
    // outlive the toolchain that built the kernels.
    println!("cargo::rustc-env=CANDLE_SYCL_ICPX_RELEASE={release}");
}

#[cfg(test)]
mod tests {
    // build scripts cannot be unit-tested through `cargo test`; compile this
    // file as a test harness instead:
    //   rustc --edition 2021 --test build.rs -o /tmp/kernels-build-tests && /tmp/kernels-build-tests
    // Only the pure parser is testable here; everything else is I/O.
    use super::parse_icpx_release;

    #[test]
    fn parses_a_published_icpx_banner() {
        // Verbatim `icpx --version` banner as published in the SPEChpc 2021
        // result hpc2021-20250916-00387 (SPEC "Compiler Version Notes" print
        // the banner), https://www.spec.org/hpc2021/results/res2025q4/
        // hpc2021-20250916-00387.txt — the same shape every oneAPI release
        // (including the 2026.1 the B70 stack runs) prints. The bare release
        // token comes first; the parenthesised build id must not win.
        let banner = "Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806)\n\
                      Target: x86_64-unknown-linux-gnu\n\
                      Thread model: posix\n\
                      InstalledDir: /opt/intel/oneapi/compiler/2025.2/bin/compiler\n\
                      Configuration file: /opt/intel/oneapi/compiler/2025.2/bin/compiler/../icpx.cfg\n";
        assert_eq!(parse_icpx_release(banner).unwrap(), "2025.2.1");
    }

    #[test]
    fn parses_the_two_component_release() {
        // `icpx --version` on the 2026.1 the SYCL stack documents names the
        // bare release the same way; docs quote it as `2026.1`.
        assert_eq!(
            parse_icpx_release("Intel(R) oneAPI DPC++/C++ Compiler 2026.1 (2026.1.0.20260708)").unwrap(),
            "2026.1"
        );
    }

    #[test]
    fn rejects_banners_without_a_release() {
        // The build must fail loudly rather than bake a guessed identity.
        assert!(parse_icpx_release("").is_err());
        assert!(parse_icpx_release("icpx: command not found").is_err());
        // A parenthesised build id alone is not a release token.
        assert!(parse_icpx_release("Intel(R) oneAPI DPC++/C++ Compiler (2026.1.0.20260708)").is_err());
        // A Level Zero driver version (1.14.37020) is not a compiler release:
        // the first component must be a four-digit year.
        assert!(parse_icpx_release("driver 1.14.37020").is_err());
    }
}
