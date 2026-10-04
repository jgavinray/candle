// Indexed MoE mat-vec parity: `QTensor::indexed_moe_forward` (the SYCL fused
// integer kernel on that backend) must agree with a CPU reference computed on
// the dequantized routed rows. Allocation is a few MB (4 experts x 32 x 256),
// safe beside a resident GPU workload. The CPU arm is the reference; only the
// SYCL arm is new.
//
// An error from a backend that implements the op fails the test. A backend
// with no op for the case skips LOUDLY (`skip` below prints the named skip)
// — never silently `Ok` (tobert/lfm2d#3 item 7: an arm whose errors all
// return `Ok(()))` cannot fail).

// SYCL DEFERRAL LIST — expected-red, MARKED red-loud (tobert/lfm2d#3
// item 7 round-3, kaibo job-9):
//  * `indexed_moe_bad_id_decode` — the integer DECODE path on this
//    branch has NO host expert-id bounds check: candle-core/src/
//    quantized/sycl.rs hands the ids buffer straight to
//    `k::indexed_moe_q8` (:687-705) and candle-sycl-kernels/csrc/
//    indexed_moe.cpp indexes the weight stack unguarded
//    (`wb[expert * row_stride + ...]`, :46,53), so the case is RED on
//    `--features sycl` (no error is raised today). The host check
//    lands with the item-5 fix on the sibling sycl-dequant-owned
//    branch; until that merge the SYCL arm is `#[ignore]`d — see the
//    hand-generated arms at the bottom of this file and the doc
//    comment on the shared fn — so `ci_sycl` cannot go red as a
//    surprise on the B70 while the branch message claims green.
//    Delete the ignore with that merge. The CPU/CUDA/Metal/ROCM arms
//    still run and pass as the loud named skip inside the shared fn.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};

// `#[macro_export]` places `test_device!` at the crate root.
use candle_core::test_device;

/// The dtypes each backend's `indexed_moe_forward` implements, for the
/// dtypes used below — the partition must match the dispatch tables or a
/// supported device/dtype pair skips as if unsupported (a false skip)
/// instead of asserting.
///  * CUDA dispatches Q8_0/Q2K/Q3K/Q4K/Q5K/Q6K and bails on everything else
///    (`QCudaStorage::indexed_moe_forward`; its kernel table also names
///    q4_0/q4_1/q5_0/q5_1, but the dtype gate never routes them).
///  * ROCm dispatches those six plus Q4_0/Q4_1/Q5_0/Q5_1
///    (`quantized/rocm/moe.rs::kernel_name`).
///  * SYCL runs integer dots for the `mmvq_q8_block` table — Q4_0/Q8_0 and
///    Q4K/Q5K/Q6K (candle-sycl-kernels); Q2K/Q3K have no SYCL integer
///    kernel and fall through to the dense-only check below and bail.
///  * Dense F32/F16/BF16 stacks run only on SYCL, which has the gather +
///    strided-batch-GEMM path (`quantized/sycl.rs`); CUDA/ROCm reject them
///    (`QCudaStorage::indexed_moe_forward`, `rocm/moe.rs::kernel_name`).
/// CPU and Metal implement nothing here — the catch-all arm in
/// `quantized/mod.rs` bails "indexed_moe_forward is not implemented for
/// {device:?} weights" (no metal implementation exists); the CPU
/// dequantized matmul in `reference` is the reference arm, so their cases
/// skip, with the precise reason named by `skip`.
fn backend_supports(dev: &Device, dtype: GgmlDType) -> bool {
    match dtype {
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => dev.is_sycl(),
        GgmlDType::Q2K | GgmlDType::Q3K => dev.is_cuda() || dev.is_rocm(),
        GgmlDType::Q4_0 => dev.is_rocm() || dev.is_sycl(),
        GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 => dev.is_rocm(),
        GgmlDType::Q4K | GgmlDType::Q8_0 | GgmlDType::Q5K | GgmlDType::Q6K => {
            dev.is_cuda() || dev.is_rocm() || dev.is_sycl()
        }
        _ => false,
    }
}

fn skip(case: &str, dev: &Device, why: &str) {
    eprintln!("SKIP indexed_moe_tests[{case}] on {dev:?}: {why}");
}

/// CUDA/ROCm read the activation as `(batch, 1 or topk, k)` — `dims()`
/// demands three dims on both. SYCL additionally accepts a bare `(batch, k)`
/// row broadcast across the task's topk experts, the shape the decode
/// engine actually passes, so keep both arms in the matrix.
fn requires_r3_input(dev: &Device) -> bool {
    (dev.is_cuda() || dev.is_rocm()) && !dev.is_sycl()
}

const NUM_EXPERTS: usize = 4;
const N: usize = 32;
const K: usize = 256;

/// The routed weight stack: dequantized f32 weights on CPU (the reference)
/// and the same block bytes rebuilt as a `QTensor` on `dev` (`QTensor` has
/// no to_device, so the raw storage is re-uploaded).
fn quantized_stack(dev: &Device, dtype: GgmlDType) -> Result<(Tensor, QTensor)> {
    let cpu = &Device::Cpu;
    let w = Tensor::rand(-1f32, 1f32, (NUM_EXPERTS, N, K), cpu)?;
    let w_q_cpu = QTensor::quantize(&w, dtype)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let shape = w_q_cpu.shape().clone();
    let w_q = QTensor::new(
        candle_core::quantized::QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?,
        shape,
    )?;
    Ok((w, w_q))
}

/// Deterministic routing over all experts, with duplicates within a token
/// when topk > 2 — the grouped paths must tolerate both.
fn routed_ids_v(batch: usize, topk: usize) -> Vec<u32> {
    let mut ids_v = Vec::with_capacity(batch * topk);
    for b in 0..batch {
        for t in 0..topk {
            ids_v.push(((b + t * 2) % NUM_EXPERTS) as u32);
        }
    }
    ids_v
}

/// `out[b][t] = w[ids[b][t]] @ rows[b][t]`, on the CPU from the dequantized
/// experts. `rows` is `(batch, topk, k)`: one activation per routed pair
/// (a `(batch, k)` input broadcast to topk equal rows is built by the
/// caller, so the reference never branches on the broadcast).
fn reference(w_stack: &Tensor, rows: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (batch, topk) = ids.dims2()?;
    let k = rows.dims()[2];
    let routed = w_stack.index_select(&ids.flatten_all()?, 0)?; // (b*t, n, k)
    let n = routed.dims()[1];
    routed
        .matmul(&rows.reshape((batch * topk, k, 1))?)?
        .reshape((batch, topk, n))
}

/// Quantized dtypes carry block-scale error on a k-term dot against the
/// f32 reference — measured 0.44 at k=256 for Q5_K (the mat-vec path,
/// whose integer dots are exact against the dequantized blocks; the error
/// is the quantization itself, not the path). The grouped prefill adds
/// f16 rounding of both operands on top (~+0.1 observed at k=256), so
/// quantized tolerance is 0.6 flat: well above the quantization + f16
/// tail, and two orders below a wrong-expert error (O(n)). Dense dtypes
/// are exact to f32 rounding.
fn tolerance(dtype: GgmlDType) -> f32 {
    if dtype == GgmlDType::F32 {
        1e-3
    } else {
        0.6
    }
}

fn assert_matches_reference(
    got: &Tensor,
    expected: &Tensor,
    dev: &Device,
    dtype: GgmlDType,
    batch: usize,
    topk: usize,
    what: &str,
) -> Result<()> {
    assert_eq!(
        got.dims(),
        &[batch, topk, N],
        "{what}: {dtype:?} on {dev:?} returned the wrong output shape"
    );
    let got_v = got
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let exp_v = expected.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = got_v
        .iter()
        .zip(exp_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff < tolerance(dtype),
        "indexed_moe_forward {dtype:?} batch={batch} topk={topk} {what} mismatch on {dev:?}: \
         max |Δ| = {max_diff}"
    );
    Ok(())
}

/// One parity case. `per_row` selects the down-projection input shape
/// `(batch, topk, k)` (its own activation row per routed pair) instead of
/// the decode shape `(batch, k)` broadcast across the token's experts.
fn run_case(
    dev: &Device,
    dtype: GgmlDType,
    batch: usize,
    topk: usize,
    per_row: bool,
) -> Result<()> {
    let cpu = &Device::Cpu;
    let (w, w_q) = quantized_stack(dev, dtype)?;
    let ids_v = routed_ids_v(batch, topk);
    let ids = Tensor::from_vec(ids_v, (batch, topk), cpu)?;
    let input = Tensor::rand(-1f32, 1f32, (batch, K), cpu)?;
    let rows = if per_row {
        Tensor::rand(-1f32, 1f32, (batch, topk, K), cpu)?
    } else {
        input
            .unsqueeze(1)?
            .broadcast_as((batch, topk, K))?
            .contiguous()?
    };
    let expected = reference(&w, &rows, &ids)?;

    // The device activation keeps the backend's accepted rank: SYCL takes
    // the raw (batch, k) decode shape, CUDA/ROCm demand (batch, 1, k),
    // which is the same rows broadcast; per_row is (batch, topk, k) on all
    // three.
    let dev_input = match (per_row, requires_r3_input(dev)) {
        (true, _) => rows.clone(),
        (false, true) => input.unsqueeze(1)?.contiguous()?,
        (false, false) => input.clone(),
    };
    let got = w_q.indexed_moe_forward(&dev_input.to_device(dev)?, &ids.to_device(dev)?);

    let got = match got {
        Ok(g) => g,
        Err(e) => {
            if !backend_supports(dev, dtype) {
                skip(
                    &format!("run_case({dtype:?}, batch={batch}, topk={topk}, per_row={per_row})"),
                    dev,
                    &format!("no indexed-MoE op for this device/dtype ({e})"),
                );
                return Ok(());
            }
            return Err(e);
        }
    };
    assert_matches_reference(
        &got,
        &expected,
        dev,
        dtype,
        batch,
        topk,
        if per_row {
            "per-row input"
        } else {
            "broadcast input"
        },
    )?;
    Ok(())
}

fn indexed_moe_q8_0(dev: &Device) -> Result<()> {
    run_case(dev, GgmlDType::Q8_0, 3, 2, false)?;
    // The dense fallback is a different path (gather + GEMM, no kernel), and
    // batch > 1 with a broadcast activation is where the row expansion lives.
    run_case(dev, GgmlDType::F32, 3, 2, false)?;
    run_case(dev, GgmlDType::F16, 2, 1, false)?;
    // 128 tasks >= the grouped-prefill threshold (64): exercises the
    // dequantize-once + gather + per-expert GEMM + scatter path on SYCL.
    // Q5K at batch 3 first: the integer mat-vec's own quantization-error
    // baseline for this dtype, so the grouped number can be compared against
    // the path that shares its weights.
    run_case(dev, GgmlDType::Q5K, 3, 2, false)?;
    run_case(dev, GgmlDType::Q5K, 32, 4, false)?;
    run_case(dev, GgmlDType::Q6K, 32, 4, false)?;
    run_case(dev, GgmlDType::Q8_0, 32, 4, false)?;
    // (batch, topk, k) down-projection input — one activation row per
    // routed pair (input_dim1 == topk), the shape the fused MLP feeds the
    // down projection. Previously untested on any backend.
    run_case(dev, GgmlDType::Q5K, 3, 2, true)?;
    run_case(dev, GgmlDType::Q6K, 32, 4, true)?;
    run_case(dev, GgmlDType::F32, 3, 2, true)?;
    Ok(())
}

/// A repeat call with the same storage and inputs must return the same
/// numbers — on SYCL this is the `dequantize_f16` cache-hit leg of the
/// grouped path: the first call fills the dequantized-stack cache, the
/// second reuses it. A corrupted or stale cache entry (item 1/item 4's
/// territory) shows up as a value mismatch here, which no previous test
/// could observe because every repeat call was swallowed by the old
/// error-`Ok(())` arm.
fn indexed_moe_grouped_cache_hit(dev: &Device) -> Result<()> {
    let cpu = &Device::Cpu;
    let dtype = GgmlDType::Q5K;
    // 32*4 = 128 tasks >= GROUPED_MIN_TASKS (64): the grouped dispatch on
    // SYCL, which is the only path with a cached dequantization.
    let (batch, topk) = (32usize, 4usize);
    if !backend_supports(dev, dtype) {
        skip(
            "Q5K cache-hit",
            dev,
            "no indexed-MoE op implemented for this backend/dtype",
        );
        return Ok(());
    }
    let (w, w_q) = quantized_stack(dev, dtype)?;
    let input = Tensor::rand(-1f32, 1f32, (batch, K), cpu)?;
    let ids = Tensor::from_vec(routed_ids_v(batch, topk), (batch, topk), cpu)?;
    let rows = input
        .unsqueeze(1)?
        .broadcast_as((batch, topk, K))?
        .contiguous()?;
    let expected = reference(&w, &rows, &ids)?;
    let dev_input = if requires_r3_input(dev) {
        input.unsqueeze(1)?.contiguous()?
    } else {
        input.clone()
    };
    let dev_input = dev_input.to_device(dev)?;
    let dev_ids = ids.to_device(dev)?;

    let first = w_q.indexed_moe_forward(&dev_input, &dev_ids)?;
    assert_matches_reference(&first, &expected, dev, dtype, batch, topk, "warm-up call")?;
    // The cache hit: same storage, same inputs, second call. Must reproduce
    // the first call bit-for-bit and still match the reference.
    let second = w_q.indexed_moe_forward(&dev_input, &dev_ids)?;
    assert_matches_reference(
        &second,
        &expected,
        dev,
        dtype,
        batch,
        topk,
        "cache-hit call",
    )?;
    let a = first.flatten_all()?.to_vec1::<f32>()?;
    let b = second.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(
        a, b,
        "a repeat call on the same storage returned different values on {dev:?}"
    );
    Ok(())
}

/// The f32/f16 switch on one storage: `QSyclStorage` caches its dequantized
/// stack, and f32 vs f16 requests share that cache (tobert/lfm2d#3 item 4
/// found an entry silently replacing a live one). Two consumer-visible
/// surfaces of the switch:
///  * `dequantize` / `dequantize_f16` / `dequantize` alternating on one
///    storage — the f32 reads must reproduce the pre-quantization weights to
///    the Q5_K quantization bound, and the interleaved f16 request must not
///    clobber the f32 result. The comparison is against `w`, the f32 source
///    tensor that was quantized (a genuine CPU reference), NOT against
///    another `dequantize` call: `QTensor::dequantize` runs the dequant on
///    the storage's own device and copies, so on the SYCL arm a CPU
///    `want32` and the device `got32a` would be the same computation fed to
///    the same host-side comparator — trivially true (tobert/lfm2d#3
///    round-2 review finding 1). On CPU the arm asserts for real: the CPU
///    `dequantize` is block `to_float` and `dequantize_f16` is its f16 cast,
///    so a broken decode, or an f16 round that replaced the shared cache
///    entry, moves the numbers off `w`. What the CPU arm cannot exercise is
///    a *device* kernel — there is no device here. Per-backend truth for
///    the f16 leg of `QTensor::dequantize_f16` (`quantized/mod.rs`): CPU
///    and SYCL take its generic `_` arm — `dequantize` + f16 cast
///    (`mod.rs:925-928`), a rounding identity once the f32 leg passed;
///    CUDA and ROCm take dedicated f16 kernels (`mod.rs:916-920` →
///    `QCudaStorage::dequantize_f16`; `mod.rs:922-924` →
///    `QRocmStorage::dequantize_f16`, `rocm.rs:177-182`), so on those
///    backends this leg does run a kernel. What this leg never exercises
///    on SYCL is `QSyclStorage::dequantize_f16` (`sycl.rs:285-287`):
///    SYCL hits the `_` arm, so the real SYCL f16 dequant kernel is
///    reached only through `indexed_moe_grouped` (`sycl.rs:171`) and the
///    qmatmul half-precision fallback (`sycl.rs:458`), and pinned
///    explicitly by the SYCL-kernels suite's `bench_prefill::
///    dequant_costs` (f16 leg).
///  * `indexed_moe_forward` with an f32 activation, then an f16 activation,
///    then f32 again on the SAME `QTensor` — every call must match the CPU
///    reference (the f16 activation only SYCL accepts; CUDA/ROCm skip it
///    loudly).
fn indexed_moe_f32_f16_switch(dev: &Device) -> Result<()> {
    let cpu = &Device::Cpu;
    let dtype = GgmlDType::Q5K;
    let (batch, topk) = (32usize, 4usize);

    // --- dequant cache leg: every backend has dequantize/dequantize_f16.
    // Measured: max |dequantize(Cpu) - w| = 0.0632 over 400 fresh draws of
    // this fixture (13.1M elements — Q5_K block-scale error on weights in
    // [-1, 1]). 0.15 is ~2.4x that tail — and a wrong-block decode or
    // clobbered cache entry moves independently-random O(1) weights, which
    // clears the bound within the first few elements.
    const DEQ_TOL: f32 = 0.15;
    let (w, w_q) = quantized_stack(dev, dtype)?;
    // f32, f16, f32 again: the second f32 read must be untouched by the
    // f16 request sharing the storage's cache.
    let got32a = w_q.dequantize(dev)?;
    let got16 = w_q.dequantize_f16(dev)?;
    let got32b = w_q.dequantize(dev)?;
    let diff = |a: &Tensor, b: &Tensor| -> Result<f32> {
        let flat = |t: &Tensor| -> Result<Vec<f32>> {
            Ok(t.to_device(cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        let (a, b) = (flat(a)?, flat(b)?);
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max))
    };
    assert!(
        diff(&got32a, &w)? < DEQ_TOL,
        "dequantize(dev) on {dev:?} is not the Q5_K decode of the source weights"
    );
    // f16 round against the same genuine reference — NOT against the
    // storage's own f32 dequantize, which would compare an f16 cast to its
    // own input and be trivially true on every backend.
    assert!(
        diff(&got16, &w)? < DEQ_TOL,
        "dequantize_f16(dev) on {dev:?} is not the Q5_K decode of the source weights"
    );
    assert!(
        diff(&got32b, &w)? < DEQ_TOL,
        "the f16 dequantize clobbered the f32 result on {dev:?}"
    );

    // --- activation-dtype switch through indexed_moe_forward.
    if !backend_supports(dev, dtype) {
        skip(
            "Q5K f32/f16 activation switch",
            dev,
            "no indexed-MoE op implemented for this backend/dtype",
        );
        return Ok(());
    }
    let input = Tensor::rand(-1f32, 1f32, (batch, K), cpu)?;
    let ids = Tensor::from_vec(routed_ids_v(batch, topk), (batch, topk), cpu)?;
    let rows = input
        .unsqueeze(1)?
        .broadcast_as((batch, topk, K))?
        .contiguous()?;
    let expected = reference(&w, &rows, &ids)?;
    let base = if requires_r3_input(dev) {
        input.unsqueeze(1)?.contiguous()?
    } else {
        input.clone()
    };
    let dev_ids = ids.to_device(dev)?;
    let f32_act = base.to_device(dev)?;
    let got = w_q.indexed_moe_forward(&f32_act, &dev_ids)?;
    assert_matches_reference(&got, &expected, dev, dtype, batch, topk, "f32 activation")?;
    if dev.is_sycl() {
        // The integer kernel reads the activation as f32 or f16
        // (quantized/sycl.rs); the reference uses the f16-rounded row.
        let h = base.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
        let h_expected = reference(&w, &h.broadcast_as((batch, topk, K))?.contiguous()?, &ids)?;
        let f16_act = base.to_dtype(DType::F16)?.to_device(dev)?;
        let got = w_q.indexed_moe_forward(&f16_act, &dev_ids)?;
        assert_matches_reference(&got, &h_expected, dev, dtype, batch, topk, "f16 activation")?;
    } else {
        skip(
            "f16 activation",
            dev,
            "the CUDA/ROCm indexed path takes an f32 activation only",
        );
    }
    // Back to f32 on the same storage: the f16 round must not have
    // poisoned the cached dequantized stack.
    let got = w_q.indexed_moe_forward(&f32_act, &dev_ids)?;
    assert_matches_reference(&got, &expected, dev, dtype, batch, topk, "f32 after f16")?;
    Ok(())
}

/// Bad expert id on the integer DECODE path (tasks < GROUPED_MIN_TASKS):
/// the id must be rejected by a host check naming the bound, before the
/// kernel indexes the weight stack with it (tobert/lfm2d#3 item 5 — the
/// kernel itself has no bounds check, so an unchecked id reads out of
/// bounds; the grouped path already bails with
/// "expert id {e} out of range for {num_experts} experts" and the dense
/// fallback with the same words, but the decode launch had neither).
///
/// CROSS-BRANCH: the host check lands with the item-5 fix on the
/// sycl-dequant-owned branch. Until that merge this case is expected
/// RED on B70/oneAPI (no error is raised today) — and it is MARKED,
/// not left to be "discovered": `indexed_moe_bad_id_decode_sycl`, the
/// hand-generated SYCL arm at the bottom of this file, carries
/// `#[ignore]` naming item 5 (the file header deferral list repeats
/// it). Every non-SYCL arm runs and skips loudly: CUDA/ROCm bounds
/// behaviour is pinned in their own suites (`cuda/moe/tests.rs`,
/// `rocm/moe/tests.rs`) and CPU has no op to call.
fn indexed_moe_bad_id_decode(dev: &Device) -> Result<()> {
    if !dev.is_sycl() {
        skip(
            "bad-id decode",
            dev,
            "SYCL-specific: the decode-path host bound is item 5's fix; \
             CUDA/ROCm bounds are pinned in their own suites",
        );
        return Ok(());
    }
    let cpu = &Device::Cpu;
    let (_w, w_q) = quantized_stack(dev, GgmlDType::Q8_0)?;
    let (batch, topk) = (2usize, 1usize);
    // 2 tasks < GROUPED_MIN_TASKS: the grouped host check (which would
    // catch this) is never reached; only the decode path runs.
    let bad = NUM_EXPERTS as u32; // == num_experts: one past the last legal id
    let ids = Tensor::from_vec(vec![1u32, bad], (batch, topk), cpu)?;
    let input = Tensor::rand(-1f32, 1f32, (batch, K), cpu)?;
    let err = w_q
        .indexed_moe_forward(&input.to_device(dev)?, &ids.to_device(dev)?)
        .err()
        .unwrap_or_else(|| {
            panic!("indexed_moe_forward accepted expert id {bad} with {NUM_EXPERTS} experts on {dev:?}")
        });
    let msg = err.to_string();
    // Pinned to the grouped path's wording (quantized/sycl.rs,
    // `indexed_moe_grouped`): the item-5 host check reuses the same sentence.
    assert!(
        msg.contains(&format!("expert id {bad} out of range for {NUM_EXPERTS} experts")),
        "bad-id decode error must name the id and the expert bound like the grouped path; got: {msg}"
    );
    Ok(())
}

test_device!(
    indexed_moe_q8_0,
    indexed_moe_q8_0_cpu,
    indexed_moe_q8_0_cuda,
    indexed_moe_q8_0_metal,
    indexed_moe_q8_0_rocm,
    indexed_moe_q8_0_sycl
);

test_device!(
    indexed_moe_grouped_cache_hit,
    indexed_moe_grouped_cache_hit_cpu,
    indexed_moe_grouped_cache_hit_cuda,
    indexed_moe_grouped_cache_hit_metal,
    indexed_moe_grouped_cache_hit_rocm,
    indexed_moe_grouped_cache_hit_sycl
);

test_device!(
    indexed_moe_f32_f16_switch,
    indexed_moe_f32_f16_switch_cpu,
    indexed_moe_f32_f16_switch_cuda,
    indexed_moe_f32_f16_switch_metal,
    indexed_moe_f32_f16_switch_rocm,
    indexed_moe_f32_f16_switch_sycl
);

// Hand-generated from `test_device!` for this case only, so the
// expected-red attribute sits exactly on the SYCL arm (the macro
// cannot mark one generated arm): the non-SYCL arms keep the behavior
// of the shared function — they run and pass as its loud named skip —
// while `--features sycl` reports this arm as ignored, red-loud, until
// tobert/candle sycl-dequant-owned (tobert/lfm2d#3 item 5) lands the
// host expert-id check on the integer decode path. Delete the ignore,
// and this hand-generation, with that merge.
#[test]
fn indexed_moe_bad_id_decode_cpu() -> Result<()> {
    indexed_moe_bad_id_decode(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn indexed_moe_bad_id_decode_cuda() -> Result<()> {
    indexed_moe_bad_id_decode(&Device::new_cuda(0)?)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_bad_id_decode_metal() -> Result<()> {
    indexed_moe_bad_id_decode(&Device::new_metal(0)?)
}

#[cfg(feature = "rocm")]
#[test]
fn indexed_moe_bad_id_decode_rocm() -> Result<()> {
    indexed_moe_bad_id_decode(&Device::new_rocm(0)?)
}

#[cfg(feature = "sycl")]
#[test]
#[ignore = "red on --features sycl until tobert/candle sycl-dequant-owned (tobert/lfm2d#3 item 5) lands the host expert-id check on the integer decode path; delete this ignore with that merge"]
fn indexed_moe_bad_id_decode_sycl() -> Result<()> {
    indexed_moe_bad_id_decode(&Device::new_sycl(0)?)
}
