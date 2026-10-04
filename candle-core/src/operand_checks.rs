//! Host-side operand validation for the fused ops dispatched to backend
//! kernels.
//!
//! The native SYCL `rms_norm`/`rope` kernels (`sycl_backend::nn_ops`) take the
//! element type from the input tensor and reinterpret the sibling buffers as
//! that same type, and `rope` indexes `cos`/`sin` with a single
//! "is-batched" flag derived from one of them. An operand pair that
//! disagrees about dtype is therefore read as garbage rather than rejected,
//! and `cos`/`sin` whose ranks disagree make the kernel run off the end of
//! the smaller buffer. Both checks mirror the operand `match`es the CPU
//! path performs (`candle_nn::ops::rms_norm`, `candle_nn::rotary_emb::rope`);
//! call them before dispatch, so that neither the native kernel nor a
//! host round-trip fallback ever sees an invalid pair.
use crate::{DType, Result};

/// All operands of `op` must share one dtype. The operand names are used in
/// the error only; the first operand is the reference.
pub fn same_dtype(op: &str, operands: &[(&str, DType)]) -> Result<()> {
    let Some(first) = operands.first() else {
        return Ok(());
    };
    let (base_name, base) = *first;
    for (name, dtype) in operands.iter().skip(1) {
        let (name, dtype) = (*name, *dtype);
        if dtype != base {
            crate::bail!(
                "dtype mismatch in {op}, {base_name} has dtype {base:?}, {name} has dtype {dtype:?}"
            );
        }
    }
    Ok(())
}

/// Whether rope's `cos`/`sin` carry a per-batch axis: rank 3 `(b, t, d/2)`
/// is batched, rank 2 `(t, d/2)` is shared across the batch. The kernels
/// index both tables with one flag, so the two must agree on the rank class.
pub fn rope_cos_batched(cos_dims: &[usize], sin_dims: &[usize]) -> Result<bool> {
    let (cos_batched, sin_batched) = (cos_dims.len() == 3, sin_dims.len() == 3);
    if cos_batched != sin_batched {
        crate::bail!(
            "rank mismatch in rope: cos has {} dims, sin has {} dims, \
             both must be rank 2 (t, d/2) or rank 3 (b, t, d/2)",
            cos_dims.len(),
            sin_dims.len()
        );
    }
    Ok(cos_batched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_dtype_accepts_a_matching_pair() {
        for dt in [DType::F16, DType::BF16, DType::F32, DType::F64, DType::U32] {
            same_dtype("rms_norm", &[("x", dt), ("alpha", dt)])
                .unwrap_or_else(|e| panic!("matching {dt:?} pair rejected: {e}"));
        }
        same_dtype(
            "rope",
            &[
                ("src", DType::F64),
                ("cos", DType::F64),
                ("sin", DType::F64),
            ],
        )
        .unwrap_or_else(|e| panic!("matching f64 triple rejected: {e}"));
    }

    #[test]
    fn same_dtype_rejects_a_mismatched_pair_naming_both_dtypes() {
        // The rms_norm bug: the kernel would take f16 from x and reinterpret
        // the f32 alpha buffer as f16. The error must name both dtypes, like
        // the CPU path's operand match.
        let err = same_dtype("rms_norm", &[("x", DType::F16), ("alpha", DType::F32)])
            .expect_err("f16 x with f32 alpha must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("F16"), "{msg}");
        assert!(msg.contains("F32"), "{msg}");
        assert!(msg.contains("rms_norm"), "{msg}");
    }

    #[test]
    fn same_dtype_rejects_a_mismatched_rope_triple_naming_every_dtype() {
        let err = same_dtype(
            "rope",
            &[
                ("src", DType::F32),
                ("cos", DType::F16),
                ("sin", DType::F16),
            ],
        )
        .expect_err("f32 src with f16 cos/sin must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("F32"), "{msg}");
        assert!(msg.contains("F16"), "{msg}");
        assert!(msg.contains("rope"), "{msg}");
    }

    #[test]
    fn rope_cos_batched_agrees_with_cpu_rank_class() {
        // rank 3 = batched (b, t, d/2), rank 2 = shared (t, d/2)
        assert!(!rope_cos_batched(&[4, 8], &[4, 8]).unwrap());
        assert!(rope_cos_batched(&[2, 4, 8], &[2, 4, 8]).unwrap());
    }

    #[test]
    fn rope_cos_batched_rejects_rank_mismatch_naming_both_ranks() {
        // The OOB bug: cos rank 3 + sin rank 2 made the kernel index the
        // half-sized sin buffer with the batched layout.
        for (cos, sin) in [
            (&[2usize, 4, 8][..], &[4, 8][..]),
            (&[4, 8][..], &[2, 4, 8][..]),
        ] {
            let err =
                rope_cos_batched(cos, sin).expect_err("rank-mismatched cos/sin must be rejected");
            let msg = err.to_string();
            assert!(msg.contains('3'), "{msg}");
            assert!(msg.contains('2'), "{msg}");
            assert!(msg.contains("rope"), "{msg}");
        }
    }
}
