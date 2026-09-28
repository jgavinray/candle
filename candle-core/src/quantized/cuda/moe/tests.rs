//! CUDA counterparts of `rocm/moe/tests.rs` for the grouped path, plus the
//! layout offsets the vector path used to ignore. Like the other tests in
//! `quantized/cuda.rs`, these need a CUDA device and fail without one.

use super::GroupedMoeRouting;
use crate::quantized::{GgmlDType, QTensor};
use crate::{Device, Result, Tensor};

/// Deterministic, spread over a couple of octaves so quantization has something
/// to lose (the same pattern as the ROCm tests).
fn ramp(len: usize, div: f32) -> Vec<f32> {
    (0..len).map(|i| (i as f32 / div).sin()).collect()
}

/// `out[b][j] = w[ids[b][j]] @ x[b][if input_dim1 == 1 { 0 } else { j }]`, on
/// the CPU from the dequantized experts.
fn reference(w: &Tensor, x: &Tensor, ids: &[u32], topk: usize) -> Result<Tensor> {
    let (batch, input_dim1, _) = x.dims3()?;
    let mut rows = Vec::with_capacity(batch * topk);
    for b in 0..batch {
        for j in 0..topk {
            let we = w.get(ids[b * topk + j] as usize)?;
            let xr = x.get(b)?.get(if input_dim1 == 1 { 0 } else { j })?;
            rows.push(we.matmul(&xr.unsqueeze(1)?)?.squeeze(1)?);
        }
    }
    Tensor::stack(&rows, 0)?.reshape((batch, topk, ()))
}

/// The q8_1 activation contract, per output row: 2% of the row's largest
/// reference magnitude (see `rocm/moe/tests.rs::check`).
fn assert_close_to_cpu(got: &[f32], want: &[f32], n: usize, what: &str) {
    for (actual, expected) in got.chunks(n).zip(want.chunks(n)) {
        let scale = expected.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() <= 0.02 * scale, "{what}: GPU {a} CPU {b}");
        }
    }
}

#[test]
fn grouped_moe_matches_cpu_and_individual_routes_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    // Partial output/input tiles, padded K stride, duplicate routes, one hot
    // expert and unused experts. Nonzero input/ID offsets catch pointer bugs.
    for dtype in [GgmlDType::Q5K, GgmlDType::Q6K] {
        for (batch, topk, input_dim1, hot) in
            [(65, 4, 1, false), (33, 4, 4, false), (67, 2, 2, true)]
        {
            let (experts, n, k) = (5, 131, 768);
            let dense =
                Tensor::from_vec(ramp(experts * n * k, 61.), (experts, n, k), &Device::Cpu)?;
            let weights = QTensor::quantize_onto(&dense, dtype, &device)?;
            let w_cpu = QTensor::quantize(&dense, dtype)?.dequantize(&Device::Cpu)?;
            let input = Tensor::from_vec(
                ramp((batch + 1) * input_dim1 * k, 43.),
                (batch + 1, input_dim1, k),
                &device,
            )?
            .narrow(0, 1, batch)?;
            let ids: Vec<u32> = (0..(batch + 1) * topk)
                .map(|i| if hot { 3 } else { ((i * 7 + i / 3) % 4) as u32 })
                .collect();
            let ids_t =
                Tensor::from_vec(ids.clone(), (batch + 1, topk), &device)?.narrow(0, 1, batch)?;
            let got = super::forward_for_test(&weights, &input, &ids_t, true)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let want = reference(&w_cpu, &input.to_device(&Device::Cpu)?, &ids[topk..], topk)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_close_to_cpu(&got, &want, n, &format!("{dtype:?} grouped"));
            // A single-route matvec quantizes the activations identically and
            // accumulates in another order: tighter than the CPU bound.
            for pair in [0, topk + 1, batch * topk - 1] {
                let x = input
                    .narrow(0, pair / topk, 1)?
                    .narrow(1, if input_dim1 == 1 { 0 } else { pair % topk }, 1)?
                    .contiguous()?;
                let id = Tensor::from_vec(vec![ids[topk + pair]], (1, 1), &device)?;
                let old = weights
                    .indexed_moe_forward(&x, &id)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let scale = old.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
                for (a, b) in got[pair * n..(pair + 1) * n].iter().zip(&old) {
                    assert!((a - b).abs() <= 2e-5 * scale, "{dtype:?}: grouped {a} matvec {b}");
                }
            }
            let again = super::forward_for_test(&weights, &input, &ids_t, true)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(got, again, "route packing must not affect arithmetic");
            let dispatched = weights
                .indexed_moe_forward(&input, &ids_t)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(got, dispatched, "public dispatch must reach grouped arithmetic");
        }
    }
    Ok(())
}

/// The vector path, below the grouped threshold, reads the rows its layouts
/// name: it used to take the storage from element 0.
#[test]
fn indexed_moe_honors_input_and_id_offsets_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (experts, n, k, batch, topk) = (4, 64, 512, 3, 2);
    for input_dim1 in [1, topk] {
        let dense = Tensor::from_vec(ramp(experts * n * k, 61.), (experts, n, k), &Device::Cpu)?;
        let weights = QTensor::quantize_onto(&dense, GgmlDType::Q5K, &device)?;
        let w_cpu = QTensor::quantize(&dense, GgmlDType::Q5K)?.dequantize(&Device::Cpu)?;
        let input = Tensor::from_vec(
            ramp((batch + 2) * input_dim1 * k, 43.),
            (batch + 2, input_dim1, k),
            &device,
        )?
        .narrow(0, 2, batch)?;
        let ids: Vec<u32> = (0..(batch + 1) * topk).map(|i| ((i * 3 + 1) % experts) as u32).collect();
        let ids_t = Tensor::from_vec(ids.clone(), (batch + 1, topk), &device)?.narrow(0, 1, batch)?;
        let got = weights.indexed_moe_forward(&input, &ids_t)?.flatten_all()?.to_vec1::<f32>()?;
        let want = reference(&w_cpu, &input.to_device(&Device::Cpu)?, &ids[topk..], topk)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close_to_cpu(&got, &want, n, &format!("vector input_dim1={input_dim1}"));
    }
    Ok(())
}

#[test]
fn grouped_moe_rejects_invalid_expert_ids_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dense = Tensor::zeros((2, 32, 256), crate::DType::F32, &device)?;
    let w = QTensor::quantize(&dense, GgmlDType::Q5K)?;
    let x = Tensor::zeros((16, 1, 256), crate::DType::F32, &device)?;
    for invalid in [2, u32::MAX] {
        let mut ids = vec![0u32; 32];
        ids[31] = invalid;
        let ids = Tensor::from_vec(ids, (16, 2), &device)?;
        let error = super::forward_for_test(&w, &x, &ids, true).unwrap_err().to_string();
        assert!(error.contains("expert id"), "{error}");
    }
    Ok(())
}

#[test]
fn grouped_moe_dispatch_keeps_decode_and_sparse_suffixes_on_matvec() {
    let mut d = super::Dims { num_experts: 32, n: 3584, k: 2048, batch: 1, topk: 4, input_dim1: 1 };
    assert!(!super::use_grouped(GgmlDType::Q5K, &d));
    d.batch = 63;
    assert!(!super::use_grouped(GgmlDType::Q5K, &d));
    d.batch = 64;
    assert!(super::use_grouped(GgmlDType::Q5K, &d));
    assert!(super::use_grouped(GgmlDType::Q6K, &d));
    assert!(!super::use_grouped(GgmlDType::Q4K, &d));
    d.k = 257;
    assert!(!super::use_grouped(GgmlDType::Q5K, &d));
}

/// One packing serves every projection of a layer and is fixed at
/// construction: what the IDs say afterwards does not reach it.
#[test]
fn prepared_routing_matches_dispatch_and_captures_ids_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (experts, n, k, batch, topk) = (4, 96, 512, 32, 2);
    let dense = Tensor::from_vec(ramp(experts * n * k, 61.), (experts, n, k), &Device::Cpu)?;
    let input = Tensor::from_vec(ramp(batch * k, 43.), (batch, 1, k), &device)?;
    let ids: Vec<u32> = (0..batch * topk).map(|i| ((i * 5 + 3) % experts) as u32).collect();
    let ids_t = Tensor::from_vec(ids, (batch, topk), &device)?;
    for dtype in [GgmlDType::Q5K, GgmlDType::Q6K] {
        let weights = QTensor::quantize_onto(&dense, dtype, &device)?;
        assert!(weights.supports_grouped_moe(batch, topk));
        let routing = GroupedMoeRouting::new(&ids_t, experts)?;
        let expected = weights.indexed_moe_forward(&input, &ids_t)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = routing.forward(&weights, &input)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(expected, actual);
        // Route everything to expert 0 in place; the packing keeps the old routes.
        ids_t.slice_set(&Tensor::zeros((batch, topk), crate::DType::U32, &device)?, 0, 0)?;
        let after = routing.forward(&weights, &input)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(expected, after, "a prepared routing must not re-read its IDs");
        let rerouted = weights.indexed_moe_forward(&input, &ids_t)?.flatten_all()?.to_vec1::<f32>()?;
        assert_ne!(expected, rerouted, "the IDs did change");
        ids_t.slice_set(
            &Tensor::from_vec(
                (0..batch * topk).map(|i| ((i * 5 + 3) % experts) as u32).collect::<Vec<_>>(),
                (batch, topk),
                &device,
            )?,
            0,
            0,
        )?;
    }
    Ok(())
}

#[test]
fn prepared_routing_rejects_bad_ids_weights_and_devices_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let ids = Tensor::zeros((32, 2), crate::DType::U32, &device)?;
    assert!(GroupedMoeRouting::new(&ids, 0).is_err());
    assert!(GroupedMoeRouting::new(&ids.to_dtype(crate::DType::F32)?, 4).is_err());
    assert!(GroupedMoeRouting::new(&Tensor::full(4u32, (32, 2), &device)?, 4).is_err());
    assert!(GroupedMoeRouting::new(&ids.transpose(0, 1)?, 4).is_err());
    assert!(GroupedMoeRouting::new(&ids.to_device(&Device::Cpu)?, 4).is_err());
    let routing = GroupedMoeRouting::new(&ids, 4)?;
    let input = Tensor::zeros((32, 1, 256), crate::DType::F32, &device)?;
    // Five experts where the routing was packed for four.
    let dense = Tensor::zeros((5, 32, 256), crate::DType::F32, &Device::Cpu)?;
    let weights = QTensor::quantize_onto(&dense, GgmlDType::Q5K, &device)?;
    assert!(routing.forward(&weights, &input).is_err());
    // An input whose batch the routing was not packed for.
    let dense = Tensor::zeros((4, 32, 256), crate::DType::F32, &Device::Cpu)?;
    let weights = QTensor::quantize_onto(&dense, GgmlDType::Q5K, &device)?;
    let short = Tensor::zeros((31, 1, 256), crate::DType::F32, &device)?;
    assert!(routing.forward(&weights, &short).is_err());
    // A format the grouped kernels do not cover.
    let q4 = QTensor::quantize_onto(&dense, GgmlDType::Q4K, &device)?;
    assert!(routing.forward(&q4, &input).is_err());
    // Weights and routing on different streams of the same GPU.
    let other = Device::Cuda(crate::CudaDevice::new_with_stream(0)?);
    let foreign = GroupedMoeRouting::new(&ids.to_device(&other)?, 4)?;
    assert!(foreign.forward(&weights, &input).is_err());
    Ok(())
}
