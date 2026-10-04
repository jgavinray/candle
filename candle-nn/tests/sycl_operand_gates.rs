//! Operand gates for the fused rms_norm / rope ops (tobert/lfm2d#3 item 6).
//!
//! The native SYCL kernels take the element type from the input and
//! reinterpret the sibling buffers as that type, so mixed-dtype operands
//! must be rejected before dispatch instead of being read as garbage.
//!
//! Two layers of pinning:
//! - The `test_device!` arms below run everywhere but pin only the
//!   PRE-EXISTING cpu_fwd dtype match (`ops.rs` rmsnorm match,
//!   `rotary_emb.rs` rope match): on a CPU device the cfg(sycl) dispatch
//!   gate is not even compiled, and those arms passed before the gate
//!   landed. They guard against regressions in the cpu operand match, not
//!   in the gate.
//! - The `#[cfg(feature = "sycl")]` tests at the bottom are the dispatch-
//!   path pins: on an Intel device with the sycl feature they reach the
//!   shim/nn_ops `same_dtype` gate (`candle::operand_checks`), and the
//!   chosen dtypes are all kernel-supported floats, so only that gate can
//!   reject the pair — pre-gate the operands reach the native kernel and
//!   the tests are red. They cannot run on this host (no oneAPI); they
//!   are compile-by-inspection here and run in the sycl CI (ci_sycl /
//!   hyper03). Message shape for the gate itself is unit-tested in
//!   `candle-core` (`operand_checks`).
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_device, DType, Device, Result, Tensor};

// --- cpu_fwd operand match (pre-existing behavior; see the module doc) ---

fn rms_norm_mixed_dtype(device: &Device) -> Result<()> {
    let xs = Tensor::ones((2, 4), DType::F16, device)?;
    let alpha = Tensor::ones(4, DType::F32, device)?;
    let err = candle_nn::ops::rms_norm(&xs, &alpha, 1e-6)
        .expect_err("f16 xs with an f32 alpha must be rejected, not reinterpreted");
    // Non-sycl devices pin cpu_fwd's catch-all; on a SYCL device the
    // shim's same_dtype gate rejects first, so the expected message
    // depends on the running device (both name the offending dtypes).
    let msg = err.to_string();
    if device.is_sycl() {
        assert!(msg.contains("dtype mismatch in rms_norm"), "{msg}");
    } else {
        assert!(msg.contains("unsupported dtype for rmsnorm"), "{msg}");
    }
    assert!(msg.contains("F16"), "{msg}");
    Ok(())
}

fn rope_mixed_dtype(device: &Device) -> Result<()> {
    // (b, h, t, d) = (1, 1, 2, 4); cos/sin are the shared rank-2 (t, d/2).
    let xs = Tensor::ones((1, 1, 2, 4), DType::F32, device)?;
    let cos = Tensor::ones((2, 2), DType::F16, device)?;
    let sin = Tensor::ones((2, 2), DType::F16, device)?;
    let err = candle_nn::rotary_emb::rope(&xs, &cos, &sin)
        .expect_err("f32 xs with f16 cos/sin must be rejected, not reinterpreted");
    let msg = err.to_string();
    if device.is_sycl() {
        assert!(msg.contains("dtype mismatch in rope"), "{msg}");
    } else {
        assert!(msg.contains("unsupported dtype for rope"), "{msg}");
    }
    assert!(msg.contains("F32"), "{msg}");
    assert!(msg.contains("F16"), "{msg}");
    Ok(())
}

fn rms_norm_matching_f16(device: &Device) -> Result<()> {
    let xs = Tensor::ones((2, 4), DType::F16, device)?;
    let alpha = Tensor::ones(4, DType::F16, device)?;
    let ys = candle_nn::ops::rms_norm(&xs, &alpha, 1e-6)?;
    assert_eq!(ys.shape(), xs.shape());
    assert_eq!(ys.dtype(), DType::F16);
    Ok(())
}

test_device!(
    rms_norm_mixed_dtype,
    rmsnorm_mix_cpu,
    rmsnorm_mix_gpu,
    rmsnorm_mix_metal,
    rmsnorm_mix_rocm,
    rmsnorm_mix_sycl
);
test_device!(
    rope_mixed_dtype,
    rope_mix_cpu,
    rope_mix_gpu,
    rope_mix_metal,
    rope_mix_rocm,
    rope_mix_sycl
);
test_device!(
    rms_norm_matching_f16,
    rmsnorm_f16_cpu,
    rmsnorm_f16_gpu,
    rmsnorm_f16_metal,
    rmsnorm_f16_rocm,
    rmsnorm_f16_sycl
);

// --- dispatch-path pins: the cfg(sycl) dtype gate -------------------------
// These run only with --features sycl on Intel hardware (ci_sycl /
// hyper03); on this mac host they are compile-by-inspection. Every dtype
// below is native-kernel-supported (float_contig accepts F16/BF16/F32),
// so on a SYCL device the ONLY thing that can reject the pair is the
// shim/nn_ops `same_dtype` gate — pre-gate, the operands reach the kernel,
// which reinterprets the sibling buffer and these tests go red.

#[cfg(feature = "sycl")]
#[test]
fn rms_norm_mixed_dtype_sycl_dispatch() -> Result<()> {
    let device = Device::new_sycl(0)?;
    let xs = Tensor::ones((2, 4), DType::F16, &device)?;
    let alpha = Tensor::ones(4, DType::F32, &device)?;
    let err = candle_nn::ops::rms_norm(&xs, &alpha, 1e-6)
        .expect_err("f16 xs with an f32 alpha must be rejected, not reinterpreted");
    // The gate's message (Error::DTypeMismatchBinaryOp) names both dtypes;
    // cpu_fwd's catch-all only names x's, so this pins the gate's path.
    let msg = err.to_string();
    assert!(msg.contains("rms_norm"), "{msg}");
    assert!(msg.contains("F16"), "{msg}");
    assert!(msg.contains("F32"), "{msg}");
    Ok(())
}

#[cfg(feature = "sycl")]
#[test]
fn rope_mixed_dtype_sycl_dispatch() -> Result<()> {
    let device = Device::new_sycl(0)?;
    let xs = Tensor::ones((1, 1, 2, 4), DType::F32, &device)?;
    let cos = Tensor::ones((2, 2), DType::F16, &device)?;
    let sin = Tensor::ones((2, 2), DType::F16, &device)?;
    let err = candle_nn::rotary_emb::rope(&xs, &cos, &sin)
        .expect_err("f32 xs with f16 cos/sin must be rejected, not reinterpreted");
    let msg = err.to_string();
    assert!(msg.contains("rope"), "{msg}");
    assert!(msg.contains("F32"), "{msg}");
    assert!(msg.contains("F16"), "{msg}");
    Ok(())
}

// --- rank semantics: cpu parity, b > 1, both directions -------------------
// cpu_fwd treats any mixed-rank cos/sin pair as the shared rank-2 layout
// without error (`unbatched_rope = cos r3 && sin r3`, rotary_emb.rs:479),
// and the native kernel now mirrors that rule exactly
// (`operand_checks::rope_cos_batched`). The pin is cpu-vs-native equality
// with b > 1: with b = 1 the batch stride (`i_cs += b_i * t * half`,
// norm.cpp:156) is never exercised, which is what let the original
// cos-only flag read a rank-2 `sin` out of bounds unnoticed for b > 1.
// The tables' batch axes carry different values, so a wrong shared/batched
// read cannot coincide with the cpu result; all values are exact binary
// fractions, so cpu and device arithmetic agree bit for bit and a tight
// tolerance suffices. Runs with the sycl feature on Intel hardware
// (ci_sycl / hyper03); compile-by-inspection elsewhere.
#[cfg(feature = "sycl")]
fn rope_rank_mismatch_matches_cpu(cos_batched: bool, sin_batched: bool) -> Result<()> {
    const B: usize = 2;
    let sycl = Device::new_sycl(0)?;
    let cpu = Device::Cpu;
    let xs = Tensor::arange(1f32, 1.0 + (B * 4) as f32, &cpu)?.reshape((B, 1, 2, 4))?;
    let table = |batched: bool, base: f32| -> Result<Tensor> {
        let shared = vec![base, base / 2.0, base + 0.25, base + 0.5];
        if batched {
            // Batch 1's row differs from batch 0, so a shared read of a
            // table the caller meant per-batch cannot agree by luck.
            let rows1: Vec<f32> = shared.iter().map(|v| *v / 2.0).collect();
            let mut values = shared.clone();
            values.extend_from_slice(&rows1);
            Tensor::from_vec(values, (B, 2, 2), &cpu)
        } else {
            Tensor::from_vec(shared, (2, 2), &cpu)
        }
    };
    let cos = table(cos_batched, 0.5)?;
    let sin = table(sin_batched, 0.125)?;
    // The cpu result is the spec: the shared read cpu_fwd performs.
    let expected = candle_nn::rotary_emb::rope(&xs, &cos, &sin)?;
    let got = candle_nn::rotary_emb::rope(
        &xs.to_device(&sycl)?,
        &cos.to_device(&sycl)?,
        &sin.to_device(&sycl)?,
    )?;
    let diff = got
        .to_device(&cpu)?
        .sub(&expected)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff < 1e-5,
        "cos {} / sin {} diverged from the cpu shared-read semantics, max diff {diff}",
        if cos_batched { "r3" } else { "r2" },
        if sin_batched { "r3" } else { "r2" },
    );
    Ok(())
}

/// cos r3 / sin r2: pre-fix the cos-only flag ran the batched stride over
/// the rank-2 sin buffer (OOB read for b > 1); round-1 rejected the pair
/// outright. Both are red against the cpu reference asserted here.
#[cfg(feature = "sycl")]
#[test]
fn rope_rank_mismatch_cos3_sin2_matches_cpu_sycl() -> Result<()> {
    rope_rank_mismatch_matches_cpu(true, false)
}

/// cos r2 / sin r3: the reverse direction. Pre-fix this was in-bounds
/// (shared read, matching cpu) but round-1's stricter predicate rejected
/// it; this test pins the parity acceptance against regression.
#[cfg(feature = "sycl")]
#[test]
fn rope_rank_mismatch_cos2_sin3_matches_cpu_sycl() -> Result<()> {
    rope_rank_mismatch_matches_cpu(false, true)
}

/// Both rank 3 at b > 1 — the genuine batched read stays intact under the
/// both-AND rule (the guard against "shared for everything" regressions).
#[cfg(feature = "sycl")]
#[test]
fn rope_rank_both3_batched_matches_cpu_sycl() -> Result<()> {
    rope_rank_mismatch_matches_cpu(true, true)
}
