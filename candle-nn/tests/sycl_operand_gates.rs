//! Operand gates for the fused rms_norm / rope ops (tobert/lfm2d#3 item 6).
//!
//! The native SYCL kernels take the element type from the input and
//! reinterpret the sibling buffers as that type, so mixed-dtype operands
//! must be rejected before dispatch instead of being read as garbage. These
//! tests pin the observable behavior through the public entry points: an
//! error on every backend, with the exact message content tested at the
//! predicate level in `candle-core` (`operand_checks`).
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_device, DType, Device, Result, Tensor};

fn rms_norm_mixed_dtype(device: &Device) -> Result<()> {
    let xs = Tensor::ones((2, 4), DType::F16, device)?;
    let alpha = Tensor::ones(4, DType::F32, device)?;
    let err = candle_nn::ops::rms_norm(&xs, &alpha, 1e-6)
        .expect_err("f16 xs with an f32 alpha must be rejected, not reinterpreted");
    let _ = err.to_string();
    Ok(())
}

fn rope_mixed_dtype(device: &Device) -> Result<()> {
    // (b, h, t, d) = (1, 1, 2, 4); cos/sin are the shared rank-2 (t, d/2).
    let xs = Tensor::ones((1, 1, 2, 4), DType::F32, device)?;
    let cos = Tensor::ones((2, 2), DType::F16, device)?;
    let sin = Tensor::ones((2, 2), DType::F16, device)?;
    let err = candle_nn::rotary_emb::rope(&xs, &cos, &sin)
        .expect_err("f32 xs with f16 cos/sin must be rejected, not reinterpreted");
    let _ = err.to_string();
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

// Rank-mismatched cos/sin (3-d vs 2-d) are accepted by the cpu path — it
// silently treats them as the shared layout — so this can only pin the
// native-path gate. Runs with the sycl feature on Intel hardware.
#[cfg(feature = "sycl")]
#[test]
fn rope_cos_sin_rank_mismatch_sycl() -> Result<()> {
    let device = Device::new_sycl(0)?;
    let xs = Tensor::ones((1, 1, 2, 4), DType::F32, &device)?;
    let cos = Tensor::ones((1, 2, 2), DType::F32, &device)?; // batched
    let sin = Tensor::ones((2, 2), DType::F32, &device)?; // shared
    let err = candle_nn::rotary_emb::rope(&xs, &cos, &sin)
        .expect_err("cos rank 3 with sin rank 2 must be rejected, not indexed batched");
    let msg = err.to_string();
    assert!(msg.contains("rope"), "{msg}");
    assert!(msg.contains('3'), "{msg}");
    assert!(msg.contains('2'), "{msg}");
    Ok(())
}
