//! Decode-shape mat-vec benchmark on the real LFM2.5-8B-A1B shapes
//! (hidden 2048, moe_intermediate 1792, 32 experts, top-4; K-quants at QK_K
//! 256). Run with `cargo test --test bench_decode --release -- --nocapture
//! --test-threads=1`. Times are wall-clock per call including pooled scratch
//! allocation, i.e. what the engine actually pays per token.
use std::sync::Arc;

use candle_sycl_kernels::*;
use half::f16;

fn bytes_of<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn bytes_of_mut<T: Copy>(s: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}

fn blk_bytes(dt: GgmlDType) -> usize {
    match dt {
        GgmlDType::Q4K => 144,
        GgmlDType::Q5K => 176,
        GgmlDType::Q6K => 210,
        GgmlDType::Q4_0 => 18,
        GgmlDType::Q8_0 => 34,
        _ => unreachable!("bench cases use the mmvq_q8 dtypes"),
    }
}

fn blk_size(dt: GgmlDType) -> usize {
    match dt {
        GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 256,
        _ => 32,
    }
}

struct Case {
    name: &'static str,
    dt: GgmlDType,
    n: usize,
    k: usize,
    batch: usize,
    topk: usize, // 0 = plain mmvq_q8, >0 = indexed_moe_q8 (32-expert stack)
}

fn launch(
    q: &Arc<Queue>,
    c: &Case,
    w: &DeviceBuffer,
    act: &DeviceBuffer,
    ids: &DeviceBuffer,
    out: &DeviceBuffer,
) -> Result<()> {
    if c.topk > 0 {
        indexed_moe_q8(
            q, c.dt, w, act, false, ids, out, false, c.n, c.k, c.batch, c.topk,
        )
        .map(|_| ())
    } else {
        mmvq_q8(q, c.dt, w, act, false, out, false, c.n, c.k, c.batch)?;
        Ok(())
    }
}

#[test]
fn decode_shapes() {
    let q = Queue::new(0).unwrap();
    let cases = [
        Case {
            name: "dense gate_up 14336x2048 Q5_K",
            dt: GgmlDType::Q5K,
            n: 14336,
            k: 2048,
            batch: 1,
            topk: 0,
        },
        Case {
            name: "dense down 2048x7168 Q5_K",
            dt: GgmlDType::Q5K,
            n: 2048,
            k: 7168,
            batch: 1,
            topk: 0,
        },
        Case {
            name: "lm_head 128256x2048 Q6_K",
            dt: GgmlDType::Q6K,
            n: 128256,
            k: 2048,
            batch: 1,
            topk: 0,
        },
        Case {
            name: "routed gate_up 3584x2048 Q5_K top4",
            dt: GgmlDType::Q5K,
            n: 3584,
            k: 2048,
            batch: 1,
            topk: 4,
        },
        Case {
            name: "routed down 2048x1792 Q5_K top4",
            dt: GgmlDType::Q5K,
            n: 2048,
            k: 1792,
            batch: 1,
            topk: 4,
        },
    ];
    for c in &cases {
        let m = if c.topk > 0 {
            c.batch * c.topk
        } else {
            c.batch
        };
        let n_exp = if c.topk > 0 { 32 } else { 1 };
        let wbytes = n_exp * c.n * (c.k / blk_size(c.dt)) * blk_bytes(c.dt);
        let w = DeviceBuffer::alloc(&q, wbytes).unwrap();
        w.copy_from_host(&vec![0x11u8; wbytes]).unwrap();
        let act = DeviceBuffer::alloc(&q, m * c.k * 4).unwrap();
        act.copy_from_host(bytes_of(&vec![0.5f32; m * c.k]))
            .unwrap();
        let out = DeviceBuffer::alloc(&q, m * c.n * 4).unwrap();
        let ids = DeviceBuffer::alloc(&q, 16).unwrap();
        if c.topk > 0 {
            ids.copy_from_host(bytes_of(&[0u32, 1, 2, 3])).unwrap();
        }
        for _ in 0..5 {
            launch(&q, c, &w, &act, &ids, &out).unwrap();
        }
        q.synchronize().unwrap();
        let iters = 200;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            launch(&q, c, &w, &act, &ids, &out).unwrap();
        }
        // Enqueue-only wall time: what the engine's host thread pays per call
        // when the device keeps up (launch-bound decode). The in-order queue
        // backs up here if the device is the slower side, in which case this
        // converges to device time instead — read it beside the total below.
        let enq_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        q.synchronize().unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let touched = if c.topk > 0 {
            c.topk * c.n * (c.k / blk_size(c.dt)) * blk_bytes(c.dt)
        } else {
            wbytes
        };
        println!(
            "{:<38} {:>8.4} ms  {:>7.1} GB/s  enqueue {:>5.1} us  (m={m})",
            c.name,
            ms,
            touched as f64 / (ms * 1e-3) / 1e9,
            enq_us
        );
    }
}

#[test]
fn q8_0_matches_reference() {
    let q = Queue::new(0).unwrap();
    let (n, k, m) = (1792usize, 2048usize, 1usize);
    let nblk = k / 32;
    let wbytes = n * nblk * 34;
    let mut wh = vec![0u8; wbytes];
    for blk_i in 0..n * nblk {
        // Fixed 1.0 fp16 scale; varied int8 payload. Arbitrary bytes in the
        // scale field can be NaN/Inf fp16, which would poison both sides.
        wh[blk_i * 34] = 0x00;
        wh[blk_i * 34 + 1] = 0x3C;
        for j in 0..32 {
            wh[blk_i * 34 + 2 + j] = ((blk_i * 32 + j) * 7 + blk_i / 34) as u8;
        }
    }
    let w = DeviceBuffer::alloc(&q, wbytes).unwrap();
    w.copy_from_host(&wh).unwrap();
    let act_h: Vec<f32> = (0..m * k)
        .map(|i| ((i % 61) as f32 - 30.0) / 16.0)
        .collect();
    let act = DeviceBuffer::alloc(&q, m * k * 4).unwrap();
    act.copy_from_host(bytes_of(&act_h)).unwrap();
    let out = DeviceBuffer::alloc(&q, m * n * 4).unwrap();
    mmvq_q8(&q, GgmlDType::Q8_0, &w, &act, false, &out, false, n, k, m).unwrap();
    q.synchronize().unwrap();
    let mut got = vec![0f32; m * n];
    out.copy_to_host(bytes_of_mut(&mut got)).unwrap();

    for row in 0..n {
        let mut acc = 0f32;
        for b in 0..nblk {
            let blk = &wh[(row * nblk + b) * 34..(row * nblk + b) * 34 + 34];
            let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let ints: Vec<i32> = blk[2..34].iter().map(|&q| q as i8 as i32).collect();
            // Reference activation quantization: BlockQ8_0 of this row's slice,
            // mirroring quantize_act (per-32 amax, round(x / d)).
            let slice = &act_h[b * 32..(b + 1) * 32];
            let amax = slice.iter().fold(0f32, |a, &x| a.max(x.abs()));
            let d_act = amax / 127.0;
            let id = if d_act != 0.0 { 1.0 / d_act } else { 0.0 };
            let q_act: Vec<i32> = slice
                .iter()
                .map(|&x| (x * id).round().clamp(-128.0, 127.0) as i32)
                .collect();
            let sum: i32 = ints.iter().zip(q_act.iter()).map(|(&a, &b)| a * b).sum();
            acc += d * d_act * sum as f32;
        }
        // One fp32 rounding per block product on each side; the device folds
        // amax in a vectorized max tree, so scales can differ from the host
        // fold by 1 ulp. Allow block-count-scaled slack, not bit equality.
        assert!(
            (acc - got[row]).abs() <= (nblk as f32) * f32::EPSILON * acc.abs() * 8.0 + 1e-4,
            "row {row}: got {} want {}",
            got[row],
            acc
        );
    }
}
