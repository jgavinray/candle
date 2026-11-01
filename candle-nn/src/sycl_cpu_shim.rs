//! Host fallback for candle-nn fused ops on the SYCL backend.
//!
//! Softmax, sigmoid, rms_norm and rope have native SYCL kernels for the
//! contiguous-float case; layer_norm has none yet. Anything the native path
//! cannot take copies to the host, runs the existing `cpu_fwd`, and copies the
//! result back. Correct, but an extra host round-trip per call, so each op
//! warns once on stderr when it happens: a run that silently leaves the device
//! hides both the cost and what actually computed the answer.
#![cfg(feature = "sycl")]
use candle::backend::{BackendDevice, BackendStorage};
use candle::{CpuStorage, Layout, Result, Shape, SyclStorage};

fn back(s: &SyclStorage, out: CpuStorage, shape: Shape) -> Result<(SyclStorage, Shape)> {
    Ok((s.device().storage_from_cpu_storage(&out)?, shape))
}

/// Say once per op that it is running on the host. See the module docs.
fn warn_once(once: &'static std::sync::Once, op: &str, reason: &str) {
    once.call_once(|| {
        eprintln!(
            "candle-nn: SYCL {op} fell back to the CPU ({reason}); \
             see candle-nn/src/sycl_cpu_shim.rs"
        );
    });
}

static SOFTMAX_ONCE: std::sync::Once = std::sync::Once::new();
static SIGMOID_ONCE: std::sync::Once = std::sync::Once::new();
static RMS_NORM_ONCE: std::sync::Once = std::sync::Once::new();
static ROPE_ONCE: std::sync::Once = std::sync::Once::new();
static LAYER_NORM_ONCE: std::sync::Once = std::sync::Once::new();

pub fn via_cpu_1(
    s: &SyclStorage,
    l: &Layout,
    f: impl FnOnce(&CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    let (out, shape) = f(&s.to_cpu_storage()?, l)?;
    back(s, out, shape)
}

#[allow(clippy::too_many_arguments)]
pub fn via_cpu_2(
    s1: &SyclStorage,
    l1: &Layout,
    s2: &SyclStorage,
    l2: &Layout,
    f: impl FnOnce(&CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    let (out, shape) = f(&s1.to_cpu_storage()?, l1, &s2.to_cpu_storage()?, l2)?;
    back(s1, out, shape)
}

#[allow(clippy::too_many_arguments)]
pub fn via_cpu_3(
    s1: &SyclStorage,
    l1: &Layout,
    s2: &SyclStorage,
    l2: &Layout,
    s3: &SyclStorage,
    l3: &Layout,
    f: impl FnOnce(
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
    ) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    let (out, shape) = f(
        &s1.to_cpu_storage()?,
        l1,
        &s2.to_cpu_storage()?,
        l2,
        &s3.to_cpu_storage()?,
        l3,
    )?;
    back(s1, out, shape)
}

/// LayerNorm has no native SYCL kernel; the host path is the only one, so it
/// says so once rather than looking like a device op.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm(
    s1: &SyclStorage,
    l1: &Layout,
    s2: &SyclStorage,
    l2: &Layout,
    s3: &SyclStorage,
    l3: &Layout,
    cpu: impl FnOnce(
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
    ) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    warn_once(&LAYER_NORM_ONCE, "layer_norm", "no native SYCL kernel yet");
    via_cpu_3(s1, l1, s2, l2, s3, l3, cpu)
}

// --- native kernels (fall back to the CPU shim when the fast path can't apply) ---

fn float_contig(s: &SyclStorage, l: &Layout) -> bool {
    use candle::backend::BackendStorage;
    use candle::DType::{BF16, F16, F32};
    // `nn_ops` densifies an offset view itself; only true non-contiguity falls back.
    l.is_contiguous() && matches!(s.dtype(), F16 | BF16 | F32)
}

pub fn softmax_last_dim(
    s: &SyclStorage,
    l: &Layout,
    cpu: impl FnOnce(&CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    if float_contig(s, l) {
        let out = candle::sycl_backend::nn_ops::softmax_last_dim(s, l)?;
        return Ok((out, l.shape().clone()));
    }
    warn_once(
        &SOFTMAX_ONCE,
        "softmax",
        "non-contiguous or non-float input",
    );
    via_cpu_1(s, l, cpu)
}

pub fn sigmoid(
    s: &SyclStorage,
    l: &Layout,
    cpu: impl FnOnce(&CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    if float_contig(s, l) {
        let out = candle::sycl_backend::nn_ops::sigmoid(s, l)?;
        return Ok((out, l.shape().clone()));
    }
    warn_once(
        &SIGMOID_ONCE,
        "sigmoid",
        "non-contiguous or non-float input",
    );
    via_cpu_1(s, l, cpu)
}

pub fn rms_norm(
    eps: f32,
    s1: &SyclStorage,
    l1: &Layout,
    s2: &SyclStorage,
    l2: &Layout,
    cpu: impl FnOnce(&CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    if float_contig(s1, l1) && float_contig(s2, l2) {
        let out = candle::sycl_backend::nn_ops::rms_norm(s1, l1, s2, l2, eps)?;
        return Ok((out, l1.shape().clone()));
    }
    warn_once(
        &RMS_NORM_ONCE,
        "rms_norm",
        "input or alpha is non-contiguous or non-float",
    );
    via_cpu_2(s1, l1, s2, l2, cpu)
}

/// `mode`: 0 interleaved (b,h,t,d), 1 half-split (b,h,t,d), 2 half-split (b,t,h,d).
#[allow(clippy::too_many_arguments)]
pub fn rope(
    mode: u32,
    s1: &SyclStorage,
    l1: &Layout,
    s2: &SyclStorage,
    l2: &Layout,
    s3: &SyclStorage,
    l3: &Layout,
    cpu: impl FnOnce(
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
    ) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    if l1.shape().dims().len() == 4
        && float_contig(s1, l1)
        && float_contig(s2, l2)
        && float_contig(s3, l3)
    {
        let out = candle::sycl_backend::nn_ops::rope(mode, s1, l1, s2, l2, s3, l3)?;
        return Ok((out, l1.shape().clone()));
    }
    warn_once(
        &ROPE_ONCE,
        "rope",
        "input must be 4-d; an operand is non-contiguous or non-float",
    );
    via_cpu_3(s1, l1, s2, l2, s3, l3, cpu)
}
