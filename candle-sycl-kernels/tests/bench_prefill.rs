//! Prefill dequant cost measurement: what the per-prefill dequantize of the
//! dense/attention weights costs at engine shapes, i.e. what the dequant
//! cache in candle-core (`QSyclStorage::dequantize*`) removes per prefill
//! after the first. Run with `cargo test --test bench_prefill --release --
//! --nocapture`. GEMM cost is unchanged by the cache; only this dequantize
//! (read quantized + write dense) disappears on cache hits.
use candle_sycl_kernels::*;

#[test]
fn dequant_costs() {
    let q = Queue::new(0).unwrap();
    let cases: [(&str, GgmlDType, usize, usize, usize); 4] = [
        // (name, dtype, n, k, bytes-per-block)
        ("gate_up 14336x2048 Q5_K", GgmlDType::Q5K, 14336, 2048, 176),
        ("down 2048x7168 Q5_K", GgmlDType::Q5K, 2048, 7168, 176),
        ("attn qkv 3072x2048 Q5_K", GgmlDType::Q5K, 3072, 2048, 176),
        (
            "lm_head 128256x2048 Q6_K",
            GgmlDType::Q6K,
            128256,
            2048,
            210,
        ),
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
            // Sentinel the dequantized output cannot legitimately hold: an
            // all-0xFF element is NaN in either f16 or f32, and dequantized
            // K-quants of an 0x11 block are finite. Element-wise (2- or
            // 4-byte) check — a lone 0xFF byte is legal inside a normal
            // f16 mantissa, so a byte scan would false-positive.
            out.copy_from_host(&vec![0xFFu8; out_len]).unwrap();
            for _ in 0..3 {
                if f16 {
                    dequantize_f16(&q, dt, &w, &out, n * k / blk).unwrap();
                } else {
                    dequantize(&q, dt, &w, &out, n * k / blk).unwrap();
                }
            }
            q.synchronize().unwrap();
            // Functional pin (item 7): prints alone could not fail. A dead
            // dequantize left the 0xFF fill (NaN element); a whole-probe
            // all-zero write is equally impossible for this input (per-block
            // products are nonzero and identical per block, so they cannot
            // cancel across the probe window).
            let esz = if f16 { 2 } else { 4 };
            let mut probe = vec![0u8; out_len.min(1 << 16)];
            out.copy_to_host(&mut probe).unwrap();
            assert!(
                !probe.chunks_exact(esz).any(|e| e.iter().all(|&b| b == 0xFF)),
                "{name} {tag}: dequantize wrote nothing (0xFF-filled element survived) — the kernel is a no-op"
            );
            assert!(
                probe.iter().any(|&b| b != 0),
                "{name} {tag}: dequantize wrote an all-zero buffer"
            );
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
            // Sanity (item 7): order-of-magnitude bounds only — a real
            // GPU dequantizes at >0.1 GB/s and no card clears 117 MB in a
            // microsecond (20 TB/s catches unit bugs); they catch breakage
            // and a hang-adjacent measurement, never hardware drift.
            let gbps = rw as f64 / (ms * 1e-3) / 1e9;
            assert!(
                ms.is_finite() && ms > 0.0,
                "{name} {tag}: {ms} ms is not a plausible per-call time"
            );
            assert!(
                gbps > 0.1 && gbps < 20_000.0,
                "{name} {tag}: {gbps} GB/s is not a plausible bandwidth"
            );
            println!("{:<28} {} {:>8.3} ms  {:>7.1} GB/s", name, tag, ms, gbps);
        }
    }
}
