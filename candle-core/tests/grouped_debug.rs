//! Stage-by-stage decomposition of the grouped MoE prefill against the CPU
//! reference, printing each stage's max|Δ| to find the diverging one. Uses
//! only public Tensor APIs; the stage-3 GEMM is expressed as an index_select
//! + matmul with the same dequantized stack the daemon path caches.
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::{GgmlDType, QStorage, QTensor};

fn reference(w: &Tensor, inp: &Tensor, ids: &Tensor) -> Result<Tensor, candle_core::Error> {
    let (batch, topk) = ids.dims2()?;
    let k = inp.dims()[1];
    let inp2 = inp.unsqueeze(1)?.broadcast_as((batch, topk, k))?.contiguous()?;
    let flat = ids.flatten_all()?;
    let routed = w.index_select(&flat, 0)?;
    let n = routed.dims()[1];
    let r = routed
        .matmul(&inp2.reshape((batch * topk, k, 1))?)?;
    println!("ref routed {:?} -> matmul {:?}", routed.shape(), r.shape());
    r.reshape((batch, topk, n))
}

#[test]
fn grouped_debug() -> Result<(), candle_core::Error> {
    let dev = Device::new_sycl(0)?;
    let cpu = &Device::Cpu;
    let (num_experts, n, k) = (4usize, 32usize, 256usize);
    let (batch, topk) = (32usize, 4usize);
    let tasks = batch * topk;
    let w = Tensor::rand(-1f32, 1f32, (num_experts, n, k), cpu)?;
    let input = Tensor::rand(-1f32, 1f32, (batch, k), cpu)?;
    let mut ids_v = Vec::with_capacity(tasks);
    for b in 0..batch { for t in 0..topk { ids_v.push(((b + t * 2) % num_experts) as u32); } }
    let ids = Tensor::from_vec(ids_v.clone(), (batch, topk), cpu)?;
    let expected = reference(&w, &input, &ids)?;

    // The daemon path: quantized stack on device, grouped = dequantize once +
    // index_select + matmul. Express exactly that with Tensor ops.
    let w_q_cpu = QTensor::quantize(&w, GgmlDType::Q5K)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let shape = w_q_cpu.shape().clone();
    let w_q = QTensor::new(QStorage::from_data(std::borrow::Cow::Owned(bytes), &dev, GgmlDType::Q5K)?, shape)?;

    // Stage 1: dequantized stack on device (the cache) vs CPU dequant.
    let wf16_dev: Tensor = w_q.dequantize_f16(&dev)?;
    let wf16_cpu = w_q_cpu.dequantize(cpu)?.to_dtype(DType::F16)?;
    let d1: f32 = (wf16_dev.to_device(cpu)? - wf16_cpu)?.abs()?.max_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    println!("stage1 dequant stack max|D| = {d1}");

    // Stage 2+3: gather task rows and matmul per expert — the grouped math.
    let flat_ids = ids.flatten_all()?;
    let act_f16 = input.to_dtype(DType::F16)?.to_device(&dev)?;
    let mut out = Vec::with_capacity(tasks);
    for t in 0..tasks {
        let e = ids_v[t];
        let w_e = wf16_dev.get(e as usize)?; // (n, k) f16
        let row = act_f16.get((t / topk) as usize)?; // (k,) f16 (broadcast)
        let r = row.reshape((1, k))?.matmul(&w_e.t()?)?.reshape((n,))?;
        out.push(r.reshape((1, n))?);
    }
    let got = Tensor::cat(&out, 0)?.to_device(cpu)?.to_dtype(DType::F32)?;
    let expected_flat = expected.reshape((tasks, n))?;
    let d3: f32 = (got - &expected_flat)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("stage2+3 gather+gemm (per-row f16) max|D| = {d3}");

    // The full grouped path through the storage API:
    let got_full = w_q.indexed_moe_forward(&input.to_device(&dev)?, &ids.to_device(&dev)?)?;
    let d4: f32 = (got_full.to_device(cpu)?.reshape((tasks, n))? - &expected_flat)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("full indexed_moe_forward max|D| = {d4}");
    assert!(d1 < 1e-2, "dequant diverged");
    assert!(d4 < 0.6 * (k as f32).sqrt(), "grouped path diverges: {d4}");
    Ok(())
}
