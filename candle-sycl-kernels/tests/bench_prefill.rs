//! Prefill dequant cost measurement: what the per-prefill dequantize of the
//! dense/attention weights costs at engine shapes, i.e. what the dequant
//! cache in candle-core (`QSyclStorage::dequantize*`) removes per prefill
//! after the first. Run with `cargo test --test bench_prefill --release --
//! --nocapture`. GEMM cost is unchanged by the cache; only this dequantize
//! (read quantized + write dense) disappears on cache hits.
use candle_sycl_kernels::*;

fn bytes_of<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[test]
fn dequant_costs() {
    let q = Queue::new(0).unwrap();
    let cases: [(&str, GgmlDType, usize, usize, usize); 4] = [
        // (name, dtype, n, k, bytes-per-block)
        ("gate_up 14336x2048 Q5_K", GgmlDType::Q5K, 14336, 2048, 176),
        ("down 2048x7168 Q5_K", GgmlDType::Q5K, 2048, 7168, 176),
        ("attn qkv 3072x2048 Q5_K", GgmlDType::Q5K, 3072, 2048, 176),
        ("lm_head 128256x2048 Q6_K", GgmlDType::Q6K, 128256, 2048, 210),
    ];
    for (name, dt, n, k, bpb) in cases {
        let blk = match dt {
            GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 256,
            _ => 32,
        };
        let wbytes = n * (k / blk) * bpb;
        let w = DeviceBuffer::alloc(&q, wbytes).unwrap();
        w.copy_from_host(&vec![0x11u8; wbytes]).unwrap();
        // f16 out (the engine's f16 pipeline) and f32 out (the f32 path).
        for (tag, f16) in [("f16", true), ("f32", false)] {
            let out_len = n * k * if f16 { 2 } else { 4 };
            let out = DeviceBuffer::alloc(&q, out_len).unwrap();
            for _ in 0..3 {
                if f16 {
                    dequantize_f16(&q, dt, &w, &out, n * k / blk).unwrap();
                } else {
                    dequantize(&q, dt, &w, &out, n * k / blk).unwrap();
                }
            }
            q.synchronize().unwrap();
            let iters = 20;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                if f16 {
                    dequantize_f16(&q, dt, &w, &out, n * k / blk).unwrap();
                } else {
                    dequantize(&q, dt, &w, &out, n * k / blk).unwrap();
                }
            }
            q.synchronize().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
            let rw = wbytes + out_len;
            println!("{:<28} {} {:>8.3} ms  {:>7.1} GB/s", name, tag, ms, rw as f64 / (ms * 1e-3) / 1e9);
        }
    }
}
