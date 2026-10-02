//! Prefill A/B: grouped path vs the integer mat-vec, real 8B shapes
//! (32 experts, gate_up 3584x2048 Q5_K, topk 4) at batch 512. Correctness is
//! owned by indexed_moe_tests; this measures device wall time. The integer
//! mat-vec timing is the batch-scaling probe's 29.8 ms at this exact shape.
use candle_core::{Device, Tensor};
use candle_core::quantized::{GgmlDType, QStorage, QTensor};

fn make_w_q(dev: &Device, bytes: &Vec<u8>, shape: candle_core::Shape) -> Result<QTensor, candle_core::Error> {
    Ok(QTensor::new(
        QStorage::from_data(std::borrow::Cow::Owned(bytes.clone()), dev, GgmlDType::Q5K)?,
        shape,
    )?)
}

#[test]
fn grouped_ab_b512() -> Result<(), candle_core::Error> {
    let dev = Device::new_sycl(0)?;
    let cpu = &Device::Cpu;
    let (num_experts, n, k) = (32usize, 3584usize, 2048usize);
    let (batch, topk) = (512usize, 4usize);
    let w = Tensor::rand(-1f32, 1f32, (num_experts, n, k), cpu)?;
    let input = Tensor::rand(-1f32, 1f32, (batch, k), cpu)?;
    let mut ids_v = Vec::with_capacity(batch * topk);
    for i in 0..batch * topk {
        ids_v.push((i % num_experts) as u32);
    }
    let ids = Tensor::from_vec(ids_v, (batch, topk), cpu)?;

    let w_q_cpu = QTensor::quantize(&w, GgmlDType::Q5K)?;
    let bytes = w_q_cpu.data()?.into_owned();
    let shape = w_q_cpu.shape().clone();
    let w_q = make_w_q(&dev, &bytes, shape.clone())?;
    let input_d = input.to_device(&dev)?;
    let ids_d = ids.to_device(&dev)?;

    // Warm cache: the dequant is amortized; steady-state grouped cost.
    for _ in 0..3 {
        w_q.indexed_moe_forward(&input_d, &ids_d)?;
    }
    dev.synchronize()?;
    let iters = 5;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        w_q.indexed_moe_forward(&input_d, &ids_d)?;
    }
    dev.synchronize()?;
    let warm_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("grouped warm-cache b512: {warm_ms:.2} ms/iter (matvec baseline 29.8 ms)");

    // Cold storage: pays the f16 dequant of the ~402 MB stack each call.
    for r in 0..2 {
        let w_cold = make_w_q(&dev, &bytes, shape.clone())?;
        w_cold.indexed_moe_forward(&input_d, &ids_d)?;
        dev.synchronize()?;
        let t = std::time::Instant::now();
        w_cold.indexed_moe_forward(&input_d, &ids_d)?;
        dev.synchronize()?;
        let cold_ms = t.elapsed().as_secs_f64() * 1e3;
        println!("grouped cold (r{r}): {cold_ms:.2} ms/iter (dequant included)");
    }
    Ok(())
}
