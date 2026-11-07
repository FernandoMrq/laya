//! A CUDA request must never be served by the CPU in silence. With the CUDA provider library
//! missing, ONNX Runtime refuses to register the CUDA execution provider; the session build has
//! to fail rather than log a warning and run on the CPU.
//!
//! The parent test copies the runtime from `ORT_DYLIB_PATH` into a temporary directory without
//! `onnxruntime_providers_cuda.dll` and re-runs itself in a child process pointed at that copy
//! (ONNX Runtime is loaded once per process). Skipped, and reported as skipped, unless the crate
//! is built with `load-dynamic` and `cuda` and `ORT_DYLIB_PATH` names a runtime.

#![cfg(all(feature = "cuda", feature = "load-dynamic"))]

use laya::laya_engine::build_session;
use laya::{LayaExecutionProvider, LayaOptions};
use std::path::{Path, PathBuf};
use std::process::Command;

const CHILD: &str = "LAYA_TEST_CUDA_WITHOUT_PROVIDER";

/// `mul_1.onnx` from the ONNX Runtime test data (MIT): `Y = X * W`, `X` float `[3, 2]`. Embedded
/// because the repository ignores `*.onnx` files.
const MUL_1: &[u8] = &[
    8, 3, 18, 6, 99, 104, 101, 110, 116, 97, 58, 112, 10, 21, 10, 1, 88, 10, 1, 87, 18, 1, 89, 26,
    5, 109, 117, 108, 95, 49, 34, 3, 77, 117, 108, 18, 8, 109, 117, 108, 32, 116, 101, 115, 116,
    42, 35, 8, 3, 8, 2, 16, 1, 34, 24, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0,
    0, 160, 64, 0, 0, 192, 64, 66, 1, 87, 90, 19, 10, 1, 88, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8,
    3, 10, 2, 8, 2, 98, 19, 10, 1, 89, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 3, 10, 2, 8, 2, 66,
    4, 10, 0, 16, 7,
];

fn write_model(dir: &Path) -> PathBuf {
    let path = dir.join("mul_1.onnx");
    std::fs::write(&path, MUL_1).unwrap();
    path
}

#[test]
fn a_cuda_session_without_the_cuda_provider_fails() {
    if let Some(dir) = std::env::var_os(CHILD) {
        let model = write_model(Path::new(&dir));
        let options = LayaOptions {
            execution_provider: LayaExecutionProvider::Cuda,
            ..LayaOptions::default()
        };
        let cpu = LayaOptions::default();

        assert!(
            build_session(&cpu, &model).is_ok(),
            "the copied runtime serves the CPU"
        );
        let error = build_session(&options, &model)
            .err()
            .expect("a CUDA session without the CUDA provider must fail, not fall back");
        eprintln!("refused as expected: {error}");
        return;
    }

    let Some(runtime) = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from) else {
        eprintln!("SKIPPED: ORT_DYLIB_PATH is not set");
        return;
    };
    let source = runtime.parent().expect("ORT_DYLIB_PATH names a file");
    let dir = tempfile::tempdir().unwrap();
    for name in ["onnxruntime.dll", "onnxruntime_providers_shared.dll"] {
        if source.join(name).exists() {
            std::fs::copy(source.join(name), dir.path().join(name)).unwrap();
        }
    }
    let copied = dir.path().join(runtime.file_name().unwrap());

    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "a_cuda_session_without_the_cuda_provider_fails",
            "--nocapture",
        ])
        .env(CHILD, dir.path())
        .env("ORT_DYLIB_PATH", &copied)
        .status()
        .unwrap();

    assert!(status.success(), "the child process failed: {status}");
}
