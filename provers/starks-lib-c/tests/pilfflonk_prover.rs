//! The prover's wrappers on what they refuse before any proof: a `provingKey/` that is not there or
//! not a pilfflonk one, a path the C side cannot take, and the GPU where there is none.
//! The prover itself is tested in C++ (`make -C pil2-stark pilfflonk_test`, `pilfflonk_prover_test.cpp`,
//! and `pilfflonk_gpu_test` against the GPU archive) and end to end in `cli/tests/pilfflonk_prove.rs`.

use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use proofman_starks_lib_c::{
    pilfflonk_gpu_available_c, pilfflonk_gpu_device_bytes_c, pilfflonk_gpu_free_bytes_c, PilFflonkDevice,
    PilFflonkErrorKind, PilFflonkProverCtx,
};

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_prover_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_missing_proving_key_is_an_io_error() {
    let dir = scratch("missing");
    let err = PilFflonkProverCtx::load(&dir.join("nothing")).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::Io, "{err}");
    assert!(err.message.contains("pilfflonk_ctx_new: globalInfo: cannot open"), "{err}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_stark_proving_key_is_a_format_error() {
    let dir = scratch("stark");
    fs::write(dir.join("pilout.globalInfo.json"), r#"{"name": "x", "backend": "stark"}"#).unwrap();
    let err = PilFflonkProverCtx::load(&dir).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::Format, "{err}");
    assert!(err.message.contains("backend: must be \"pilfflonk\""), "{err}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_path_with_a_nul_byte_is_refused_before_the_call() {
    let err = PilFflonkProverCtx::load(Path::new(OsStr::from_bytes(b"a\0b"))).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::InvalidArgument, "{err}");
    assert!(err.message.contains("pilfflonk_ctx_new") && err.message.contains("NUL"), "{err}");
}

#[test]
fn the_cpu_device_is_load() {
    let dir = scratch("cpu");
    let err = PilFflonkProverCtx::load_on(&dir.join("nothing"), PilFflonkDevice::Cpu).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::Io, "{err}");
    assert!(err.message.contains("pilfflonk_ctx_new_on: globalInfo: cannot open"), "{err}");
    assert_eq!(PilFflonkDevice::default(), PilFflonkDevice::Cpu);
    fs::remove_dir_all(&dir).unwrap();
}

/// Without a GPU (a library built `cpu-only`, or a machine without one), the GPU is refused before
/// any file is read, saying why; with one, the key is read as on the CPU. The library says which.
#[test]
fn the_gpu_is_refused_without_one() {
    let dir = scratch("gpu");
    let err = PilFflonkProverCtx::load_on(&dir.join("nothing"), PilFflonkDevice::Gpu).unwrap_err();
    if pilfflonk_gpu_available_c() {
        assert_eq!(err.kind, PilFflonkErrorKind::Io, "{err}");
    } else {
        assert_eq!(err.kind, PilFflonkErrorKind::InvalidArgument, "{err}");
        assert!(err.message.contains("pilfflonk_ctx_new_on: ProvingKey::load: no GPU"), "{err}");
        // build.rs's choice: the GPU archive if it found nvcc and `cpu-only` is off.
        let why = if env!("STARKS_BUILD_MODE") == "GPU" { "CUDA sees no device" } else { "built without it" };
        assert!(err.message.contains(why), "{err}");
    }
    let err = PilFflonkProverCtx::load_on(Path::new(OsStr::from_bytes(b"a\0b")), PilFflonkDevice::Gpu).unwrap_err();
    assert!(err.message.contains("pilfflonk_ctx_new_on") && err.message.contains("NUL"), "{err}");
    fs::remove_dir_all(&dir).unwrap();
}

/// A key on a device buffer of the caller's, the device memory a key needs and the device's free
/// memory: as the GPU, refused without one before any file is read, saying why; with one, the key is
/// read as on the CPU. A host address stands for the buffer, which nothing reads, since the key is
/// refused first; a null one is refused.
#[test]
fn a_device_buffer_and_the_device_bytes_are_refused_without_a_gpu() {
    let dir = scratch("gpu_buffer");
    let missing = dir.join("nothing");
    let mut stand_in = 0u64;
    let buffer: *mut std::ffi::c_void = (&mut stand_in as *mut u64).cast();
    // SAFETY: no key is loaded, so the buffer is never used.
    let on_buffer = unsafe { PilFflonkProverCtx::load_on_device_buffer(&missing, buffer, 8) }.unwrap_err();
    let bytes = pilfflonk_gpu_device_bytes_c(&missing).unwrap_err();
    if pilfflonk_gpu_available_c() {
        assert_eq!(on_buffer.kind, PilFflonkErrorKind::Io, "{on_buffer}");
        assert_eq!(bytes.kind, PilFflonkErrorKind::Io, "{bytes}");
    } else {
        assert_eq!(on_buffer.kind, PilFflonkErrorKind::InvalidArgument, "{on_buffer}");
        assert!(
            on_buffer.message.contains("pilfflonk_ctx_new_on_device_buffer: ProvingKey::load: no GPU"),
            "{on_buffer}"
        );
        assert_eq!(bytes.kind, PilFflonkErrorKind::InvalidArgument, "{bytes}");
        assert!(
            bytes.message.contains("pilfflonk_gpu_device_bytes: ProvingKey::requiredDeviceBytes: no GPU"),
            "{bytes}"
        );
    }
    // The device's free memory, as a load checks it: refused without a GPU, as the device bytes.
    match pilfflonk_gpu_free_bytes_c() {
        Ok(free) => assert!(pilfflonk_gpu_available_c() && free > 0, "{free} bytes free"),
        Err(err) => {
            assert!(!pilfflonk_gpu_available_c(), "{err}");
            assert_eq!(err.kind, PilFflonkErrorKind::InvalidArgument, "{err}");
            assert!(err.message.contains("pilfflonk_gpu_free_bytes: ProvingKey::freeDeviceBytes: no GPU"), "{err}");
        }
    }
    // SAFETY: refused before anything is read.
    let null = unsafe { PilFflonkProverCtx::load_on_device_buffer(&missing, std::ptr::null_mut(), 8) }.unwrap_err();
    assert_eq!(null.kind, PilFflonkErrorKind::InvalidArgument, "{null}");
    assert!(null.message.contains("device_buffer is NULL"), "{null}");
    fs::remove_dir_all(&dir).unwrap();
}
