// Indexed MoE mat-vec parity: `QTensor::indexed_moe_forward` (the SYCL fused
// integer kernel on that backend) must agree with a CPU reference computed on
// the dequantized routed rows. Allocation is a few MB (4 experts x 32 x 256),
// safe beside a resident GPU workload. The CPU arm is the reference; only the
// SYCL arm is new.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result, Tensor};

// `#[macro_export]` places `test_device!` at the crate root.
use candle_core::test_device;

fn reference(
    w_stack: &Tensor, // (num_experts, n, k) dequantized
    input: &Tensor,   // (batch, k)
    ids: &Tensor,     // (batch, topk) u32
) -> Result<Tensor> {
    let (batch, topk) = ids.dims2()?;
    let k = input.dims()[1];
    // (batch, 1, k) broadcasts to (batch, topk, k).
    let inp = input
        .unsqueeze(1)?
        .broadcast_as((batch, topk, k))?
        .contiguous()?; // (b,t,k)
    let flat = ids.flatten_all()?;
    let routed = w_stack.index_select(&flat, 0)?; // (b*t, n, k)
    let n = routed.dims()[1];
    routed
        .matmul(&inp.reshape((batch * topk, k, 1))?)?
        .reshape((batch, topk, n))
}

fn indexed_moe_q8_0(dev: &Device) -> Result<()> {
    run_case(dev, GgmlDType::Q8_0, 3, 2)?;
    // The dense fallback is a different path (gather + GEMM, no kernel), and
    // batch > 1 with a broadcast activation is where the row expansion lives.
    run_case(dev, GgmlDType::F32, 3, 2)?;
    run_case(dev, GgmlDType::F16, 2, 1)?;
    // 128 tasks >= the grouped-prefill threshold (64): exercises the
    // dequantize-once + gather + per-expert GEMM + scatter path on SYCL.
    // Q5K at batch 3 first: the integer mat-vec's own quantization-error
    // baseline for this dtype, so the grouped number can be compared against
    // the path that shares its weights.
    run_case(dev, GgmlDType::Q5K, 3, 2)?;
    run_case(dev, GgmlDType::Q5K, 32, 4)?;
    run_case(dev, GgmlDType::Q6K, 32, 4)?;
    run_case(dev, GgmlDType::Q8_0, 32, 4)?;
    Ok(())
}

fn run_case(dev: &Device, dtype: GgmlDType, batch: usize, topk: usize) -> Result<()> {
    let cpu = &Device::Cpu;
    let num_experts = 4usize;
    let (n, k) = (32usize, 256usize);

    let w = Tensor::rand(-1f32, 1f32, (num_experts, n, k), cpu)?;
    let input = Tensor::rand(-1f32, 1f32, (batch, k), cpu)?;
    let mut ids_v = Vec::with_capacity(batch * topk);
    for b in 0..batch {
        for t in 0..topk {
            ids_v.push(((b + t * 2) % num_experts) as u32);
        }
    }
    let ids = Tensor::from_vec(ids_v, (batch, topk), cpu)?;

    let expected = reference(&w, &input, &ids)?;

    // Quantize on CPU, then rebuild the QTensor on the device under test from
    // the same raw block bytes (`QTensor` has no to_device).
    let w_q_cpu = QTensor::quantize(&w, dtype)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let shape = w_q_cpu.shape().clone();
    let w_q = QTensor::new(
        candle_core::quantized::QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype)?,
        shape,
    )?;

    let got = w_q.indexed_moe_forward(&input.to_device(dev)?, &ids.to_device(dev)?);

    let got = match got {
        Ok(g) => g,
        Err(e) => {
            if dev.is_sycl() {
                return Err(e);
            }
            // Backends without the op are skipped, not failed: the CPU arm
            // above is the reference.
            return Ok(());
        }
    };

    let got_v = got.to_device(cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let exp_v = expected.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = got_v
        .iter()
        .zip(exp_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // Quantized dtypes carry block-scale error on a k-term dot against the
    // f32 reference — measured 0.44 at k=256 for Q5_K (the mat-vec path,
    // whose integer dots are exact against the dequantized blocks; the error
    // is the quantization itself, not the path). The grouped prefill adds
    // f16 rounding of both operands on top (~+0.1 observed at k=256), so
    // quantized tolerance is 0.6 flat: well above the quantization + f16
    // tail, and two orders below a wrong-expert error (O(n)). Dense dtypes
    // are exact to f32 rounding.
    let tol = if dtype == GgmlDType::F32 { 1e-3 } else { 0.6 };
    assert!(
        max_diff < tol,
        "indexed_moe_forward {dtype:?} batch={batch} topk={topk} mismatch on {dev:?}: \
         max |Δ| = {max_diff}"
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
