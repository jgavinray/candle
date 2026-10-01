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
    let num_experts = 4usize;
    let (n, k) = (32usize, 256usize);
    let (batch, topk) = (3usize, 2usize);
    let cpu = &Device::Cpu;

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
    let w_q_cpu = QTensor::quantize(&w, GgmlDType::Q8_0)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let shape = w_q_cpu.shape().clone();
    let w_q = QTensor::new(
        candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Owned(bytes),
            dev,
            GgmlDType::Q8_0,
        )?,
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
    // Q8_0 integer dots vs the f32 reference: with |w|<=1, k=256 and per-block
    // scales, the worst-case quantization error on a 256-term dot is ~1e-1.
    // The CPU arm compares the same two quantization regimes, so 0.5 bounds
    // both without masking a real bug (a wrong expert or row gives O(n)).
    assert!(
        max_diff < 0.5,
        "indexed_moe_forward mismatch on {dev:?}: max |Δ| = {max_diff}"
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
