//! Host-side operand validation for the fused ops dispatched to backend
//! kernels.
//!
//! The native SYCL `rms_norm`/`rope` kernels (`sycl_backend::nn_ops`) take the
//! element type from the input tensor and reinterpret the sibling buffers as
//! that same type, and `rope` indexes `cos`/`sin` with a single
//! "is-batched" flag shared by both tables. An operand pair that disagrees
//! about dtype is therefore read as garbage rather than rejected, and a
//! batched flag over a rank-2 `(t, d/2)` table makes the kernel run off the
//! end of that buffer (`i_cs += b_i * t * half`, `csrc/norm.cpp`). Both
//! checks mirror the rules the CPU path applies — the operand `match`es in
//! `candle_nn::ops::rms_norm` / `candle_nn::rotary_emb` and the
//! `unbatched_rope = cos rank 3 && sin rank 3` rule in `RotaryEmb::cpu_fwd`;
//! call them before dispatch, so that neither the native kernel nor a host
//! round-trip fallback ever sees an invalid pair.
use crate::{DType, Error, Result};

/// All operands of `op` must share one dtype. `operands` are in call order
/// (rms_norm: `x`, `alpha`; rope: `src`, `cos`, `sin`) and the first is the
/// reference. Reuses [`Error::DTypeMismatchBinaryOp`] — the same shape
/// `Storage::same_dtype` raises for the core binary ops — naming the
/// reference dtype as `lhs` and the first offender as `rhs`, so downstream
/// matchers see one dtype-mismatch style across the crate.
pub fn same_dtype(op: &'static str, operands: &[DType]) -> Result<()> {
    let Some(&base) = operands.first() else {
        return Ok(());
    };
    for &dtype in operands.iter().skip(1) {
        if dtype != base {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: base,
                rhs: dtype,
                op,
            }
            .bt());
        }
    }
    Ok(())
}

/// Whether rope's `cos`/`sin` carry a per-batch axis. The kernels index both
/// tables with one flag, so the flag may be batched only when **both**
/// tables are rank 3 `(b, t, d/2)`; every other combination — including the
/// rank-2/rank-3 mixes — reads both tables as the shared rank-2 layout
/// `(t, d/2)`. This is the exact rule of the CPU path (`let unbatched_rope =
/// l_cos.dims().len() == 3 && l_sin.dims().len() == 3` in `RotaryEmb::cpu_fwd`,
/// rotary_emb.rs:479), and it is what closes the original out-of-bounds:
/// deriving the flag from `cos` alone sent a rank-2 `sin` through the
/// batched stride (norm.cpp:156-157) for `b > 1`. Mixed ranks are not an
/// error — the shared read is in-bounds for both buffers and is also what
/// the cpu round-trip fallback produces, so the native and fallback routes
/// agree on every accepted rank combination.
pub fn rope_cos_batched(cos_dims: &[usize], sin_dims: &[usize]) -> bool {
    cos_dims.len() == 3 && sin_dims.len() == 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_dtype_accepts_a_matching_pair() {
        for dt in [DType::F16, DType::BF16, DType::F32, DType::F64, DType::U32] {
            same_dtype("rms_norm", &[dt, dt])
                .unwrap_or_else(|e| panic!("matching {dt:?} pair rejected: {e}"));
        }
        same_dtype("rope", &[DType::F64, DType::F64, DType::F64])
            .unwrap_or_else(|e| panic!("matching f64 triple rejected: {e}"));
    }

    #[test]
    fn same_dtype_rejects_a_mismatched_pair_naming_both_dtypes() {
        // The rms_norm bug: the kernel would take f16 from x and reinterpret
        // the f32 alpha buffer as f16. The error must name both dtypes in
        // the crate's DTypeMismatchBinaryOp shape, like Storage::same_dtype.
        let err = same_dtype("rms_norm", &[DType::F16, DType::F32])
            .expect_err("f16 x with f32 alpha must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("dtype mismatch in rms_norm"), "{msg}");
        assert!(msg.contains("lhs: F16"), "{msg}");
        assert!(msg.contains("rhs: F32"), "{msg}");
    }

    #[test]
    fn same_dtype_rejects_a_mismatched_rope_triple_naming_every_dtype() {
        let err = same_dtype("rope", &[DType::F32, DType::F16, DType::F16])
            .expect_err("f32 src with f16 cos/sin must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("dtype mismatch in rope"), "{msg}");
        assert!(msg.contains("lhs: F32"), "{msg}");
        assert!(msg.contains("rhs: F16"), "{msg}");
    }

    #[test]
    fn rope_cos_batched_matches_the_cpu_rank_rule() {
        // Both rank 3 = batched; everything else — including the mixed
        // rank combinations — is the shared rank-2 read, mirroring cpu_fwd's
        // `unbatched_rope = cos r3 && sin r3` formula (rotary_emb.rs:479).
        assert!(!rope_cos_batched(&[4, 8], &[4, 8]));
        assert!(rope_cos_batched(&[2, 4, 8], &[2, 4, 8]));
        // The round-1 gate rejected these two; the cpu path (and now the
        // native path, sharing the fallback's semantics) reads them shared.
        assert!(!rope_cos_batched(&[2, 4, 8], &[4, 8]));
        assert!(!rope_cos_batched(&[4, 8], &[2, 4, 8]));
        // Degenerate ranks fall to shared, as the cpu's == 3 checks do.
        assert!(!rope_cos_batched(&[8], &[8]));
        assert!(!rope_cos_batched(&[2, 2, 4, 8], &[2, 2, 4, 8]));
    }

    #[test]
    fn rope_cos_batched_batched_flag_never_indexes_a_rank2_table() {
        // Mutation pin for the OOB fix (norm.cpp:156-157): the batched
        // stride may only run when both tables carry the batch axis. If the
        // rule ever degrades to "cos says so" (|| or cos-only), the mixed
        // rows below flip to true and this fails.
        for (cos, sin) in [
            (&[2usize, 4, 8][..], &[4, 8][..]),
            (&[4, 8][..], &[2, 4, 8][..]),
        ] {
            assert!(
                !rope_cos_batched(cos, sin),
                "rank-mixed cos {cos:?} / sin {sin:?} must read shared"
            );
        }
    }
}
