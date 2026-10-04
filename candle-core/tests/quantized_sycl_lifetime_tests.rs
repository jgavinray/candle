// Regression pins for tobert/lfm2d#3 items 1 and 5 on the SYCL backend.
//
// Item 1 (use-after-free): `QTensor::dequantize` used to hand out a view of
// the per-storage dequantize cache; `quantized_lfm2_moe` kept the result as
// a model tensor, and when the temporary QTensor dropped, the cache entry
// dropped with it, the buffer went back to the queue's exact-size pool, and
// the next same-sized allocation re-handed the pointer the kept tensor was
// still reading (quantized_lfm2_moe.rs:883 / :925). The public API now owns
// its result. Test A reproduces the verdict's minimal-fix step 5: keep a
// dequantize() result across the QTensor's drop, force same-sized pool
// re-hands, and re-read.
//
// Item 5 (OOB expert ids): the integer indexed-MoE kernel indexes the
// weight stack with raw ids and no bounds check; the decode launch path now
// validates on the host like the grouped path always did. Test B asserts
// the error names the bad id and the bound.
//
// The device tests need an Intel GPU: they are `#[cfg(feature = "sycl")]`
// with a runtime `Device::new_sycl` guard and skip loudly when no device is
// present (the pattern of candle-transformers/tests/lfm2_moe_sycl.rs, minus
// the env-var gate — a missing device is always a skip, never a failure).
// Without the sycl feature every test here still compiles and prints a skip
// note, so `cargo test -p candle-core` stays green on machines with no
// oneAPI. Run on the B70 with `cargo test -p candle-core --features sycl
// --test quantized_sycl_lifetime_tests`.


use candle_core::quantized::{
    check_expert_ids, dequant_cache_decision, DequantCacheDecision, GgmlDType, QStorage, QTensor,
};
use candle_core::{Device, Result, Tensor};

/// The pool bucket is keyed by exact byte length, so re-hands need
/// allocations of exactly the dequantized buffer's size.
const N: usize = 32;
const K: usize = 256;

#[cfg(feature = "sycl")]
fn try_sycl(test: &str) -> Option<Device> {
    match Device::new_sycl(0) {
        Ok(dev) => Some(dev),
        Err(err) => {
            eprintln!("SKIP({test}): no SYCL device at ordinal 0: {err}");
            None
        }
    }
}

#[cfg(not(feature = "sycl"))]
fn try_sycl(test: &str) -> Option<Device> {
    eprintln!(
        "SKIP({test}): candle-core is built without the sycl feature; \
         rebuild with --features sycl on a oneAPI host"
    );
    None
}

#[test]
fn dequant_cache_policy_from_the_consumer_side() {
    // The transitions the SYCL cache must make, asserted through the public
    // surface so the fix cannot quietly regress them (see the exhaustive
    // matrix in quantized/mod.rs's unit tests): only an exact-key match may
    // reuse an entry, and every key change retires — never drops — the old
    // owner (tobert/lfm2d#3 item 1).
    assert_eq!(
        dequant_cache_decision(Some((N * K, false)), (N * K, false)),
        DequantCacheDecision::Keep
    );
    assert_eq!(
        dequant_cache_decision(Some((N * K, false)), (N * K, true)),
        DequantCacheDecision::EvictOld
    );
    assert_eq!(
        dequant_cache_decision(Some((N * K, true)), (N * K, false)),
        DequantCacheDecision::EvictOld
    );
    assert_eq!(
        dequant_cache_decision(None, (N * K, false)),
        DequantCacheDecision::Insert
    );
}

#[test]
fn expert_id_gate_names_the_bad_id_and_the_bound() {
    // The host gate behind the integer decode launch (tobert/lfm2d#3 item
    // 5), observable without a device.
    check_expert_ids(&[0, 3, 3, 1], 4).unwrap();
    let msg = check_expert_ids(&[0, 4], 4).unwrap_err().to_string();
    assert!(
        msg.contains("expert id 4 out of range for 4 experts"),
        "the bad-id error must name the id and the bound, got: {msg}"
    );
}

/// Item 1: a tensor kept from `QTensor::dequantize` must survive the QTensor
/// dropping and survive the exact-size pool handing its old buffer to new
/// allocations of the same size.
#[test]
fn dequantized_tensor_survives_the_qtensor_and_exact_size_reallocations() -> Result<()> {
    const TEST: &str = "dequantized_tensor_survives_the_qtensor_and_exact_size_reallocations";
    let cpu = &Device::Cpu;
    let w = Tensor::rand(-1f32, 1f32, (N, K), cpu)?;
    let w_q_cpu = QTensor::quantize(&w, GgmlDType::Q5K)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let shape = w_q_cpu.shape().clone();
    // CPU dequantize of the same block bytes is the reference: both sides
    // apply the same integer scales, so any drift is rounding, while a
    // recycled-buffer corruption moves elements by O(1).
    let expected = w_q_cpu.dequantize(cpu)?.flatten_all()?.to_vec1::<f32>()?;

    let Some(dev) = try_sycl(TEST) else {
        return Ok(());
    };

    // The quantized_lfm2_moe load shape: dequantize through the public API
    // from a temporary QTensor and keep the result past the QTensor's drop
    // (quantized_lfm2_moe.rs:883 keeps exp_probs_b.bias exactly this way).
    let kept = {
        let temp = QTensor::new(
            QStorage::from_data(std::borrow::Cow::Owned(bytes.clone()), &dev, GgmlDType::Q5K)?,
            shape,
        )?;
        temp.dequantize(&dev)?
    }; // temp — and with it its QSyclStorage and the dequantize cache — drops here.

    // Force same-size pool re-hands: the dequantized buffer is N*K f32s, and
    // the queue's allocation cache buckets by exact byte length, so these
    // allocations pop whatever the drop gave back. Write a sentinel that
    // quantized values (all in [-1, 1]) can never reach: if `kept` aliases a
    // re-handed buffer, the read-back moves toward 7.25.
    for _ in 0..8 {
        let poison = Tensor::full(7.25f32, N * K, &dev)?;
        drop(poison);
    }

    let got = kept.to_device(cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = got
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // 1e-3: far above GPU-vs-CPU dequantize rounding, far below the ~6.25
    // jump a poisoned element makes (sentinel vs values bounded by 1).
    assert!(
        max_diff <= 1e-3,
        "the kept dequantize() result changed after the temporary QTensor \
         dropped and same-sized buffers were recycled through the pool \
         (tobert/lfm2d#3 item 1): max |Δ| = {max_diff}"
    );
    Ok(())
}

/// Item 5: out-of-range expert ids on the integer decode path must be
/// rejected by the host before the kernel launches, naming id and bound.
#[test]
fn indexed_moe_decode_rejects_out_of_range_expert_ids() -> Result<()> {
    const TEST: &str = "indexed_moe_decode_rejects_out_of_range_expert_ids";
    let cpu = &Device::Cpu;
    let Some(dev) = try_sycl(TEST) else {
        return Ok(());
    };
    let (num_experts, n, k) = (4usize, 32usize, 256usize);

    // Q8_0 so the integer mat-vec kernel serves the call (batch*topk = 2
    // tasks < the grouped threshold, k % 32 == 0): exactly the path that
    // pre-fix launched with unvalidated ids and read the stack out of
    // bounds (tobert/lfm2d#3 item 5).
    let w = Tensor::rand(-1f32, 1f32, (num_experts, n, k), cpu)?;
    let w_q_cpu = QTensor::quantize(&w, GgmlDType::Q8_0)?;
    let w_q = QTensor::new(
        QStorage::from_data(
            std::borrow::Cow::Owned(w_q_cpu.data()?.into_owned()),
            &dev,
            GgmlDType::Q8_0,
        )?,
        w_q_cpu.shape().clone(),
    )?;
    let input = Tensor::rand(-1f32, 1f32, (1, k), cpu)?.to_device(&dev)?;

    let bad_ids = Tensor::from_vec(vec![0u32, 9], (1, 2), cpu)?.to_device(&dev)?;
    let msg = match w_q.indexed_moe_forward(&input, &bad_ids) {
        Ok(_) => candle_core::bail!(
            "expert id 9 out of range for {num_experts} experts was accepted by \
             indexed_moe_forward (tobert/lfm2d#3 item 5)"
        ),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("expert id 9"),
        "the bad-id error must name the offending id, got: {msg}"
    );
    assert!(
        msg.contains("4 experts"),
        "the bad-id error must name the bound, got: {msg}"
    );

    // In-range ids on the same stack must still run.
    let ok_ids = Tensor::from_vec(vec![0u32, 3], (1, 2), cpu)?.to_device(&dev)?;
    w_q.indexed_moe_forward(&input, &ok_ids)?;
    Ok(())
}
