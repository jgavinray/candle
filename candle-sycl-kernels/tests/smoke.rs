// Requires a real Intel GPU + oneAPI. Run inside the `candle-sycl-dev` container.
use candle_sycl_kernels::*;

fn bytes_of<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn bytes_of_mut<T: Copy>(s: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}

#[test]
fn device_opens() {
    let q = Queue::new(0).unwrap();
    let info = q.device_info().unwrap();
    println!(
        "{} | {} CU | {:.1} GB | integrated={} fp16={}",
        info.name,
        info.max_compute_units,
        info.global_mem_bytes as f64 / 1e9,
        info.is_integrated,
        info.supports_fp16
    );
    assert!(info.max_compute_units > 0);
}

#[test]
fn roundtrip_and_affine() {
    let q = Queue::new(0).unwrap();
    let host: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let buf = DeviceBuffer::alloc(&q, host.len() * 4).unwrap();
    buf.copy_from_host(bytes_of(&host)).unwrap();

    let out = DeviceBuffer::alloc(&q, host.len() * 4).unwrap();
    affine(
        &q,
        SyclDType::F32,
        &Layout::dense(),
        &buf,
        &out,
        host.len(),
        3.0,
        1.0,
    )
    .unwrap();

    let mut got = vec![0f32; 16];
    out.copy_to_host(bytes_of_mut(&mut got)).unwrap();
    for (i, v) in got.iter().enumerate() {
        assert_eq!(*v, i as f32 * 3.0 + 1.0, "idx {i}");
    }
}

#[test]
fn unary_exp_and_binary_add() {
    let q = Queue::new(0).unwrap();
    let a: Vec<f32> = vec![0.0, 1.0, 2.0];
    let b: Vec<f32> = vec![10.0, 20.0, 30.0];
    let da = DeviceBuffer::alloc(&q, 12).unwrap();
    let db = DeviceBuffer::alloc(&q, 12).unwrap();
    let dc = DeviceBuffer::alloc(&q, 12).unwrap();
    da.copy_from_host(bytes_of(&a)).unwrap();
    db.copy_from_host(bytes_of(&b)).unwrap();

    binary(
        &q,
        BinaryOp::Add,
        SyclDType::F32,
        &Layout::dense(),
        &da,
        &Layout::dense(),
        &db,
        &dc,
        3,
    )
    .unwrap();
    let mut c = vec![0f32; 3];
    dc.copy_to_host(bytes_of_mut(&mut c)).unwrap();
    assert_eq!(c, vec![10.0, 21.0, 32.0]);

    unary(
        &q,
        UnaryOp::Exp,
        SyclDType::F32,
        &Layout::dense(),
        &da,
        &dc,
        3,
    )
    .unwrap();
    dc.copy_to_host(bytes_of_mut(&mut c)).unwrap();
    assert!((c[0] - 1.0).abs() < 1e-5 && (c[1] - std::f32::consts::E).abs() < 1e-4);
}

#[test]
fn gemm_f32_identity() {
    let q = Queue::new(0).unwrap();
    // 2x3 @ 3x2 = 2x2, row-major, all ones -> each entry = 3
    let a = vec![1f32; 6];
    let b = vec![1f32; 6];
    let da = DeviceBuffer::alloc(&q, 24).unwrap();
    let db = DeviceBuffer::alloc(&q, 24).unwrap();
    let dc = DeviceBuffer::alloc(&q, 16).unwrap();
    da.copy_from_host(bytes_of(&a)).unwrap();
    db.copy_from_host(bytes_of(&b)).unwrap();
    gemm(
        &q,
        SyclDType::F32,
        false,
        false,
        2,
        2,
        3,
        1.0,
        0.0,
        &da,
        &db,
        &dc,
        1,
        0,
        0,
        0,
        0,
        0,
    )
    .unwrap();
    let mut c = vec![0f32; 4];
    dc.copy_to_host(bytes_of_mut(&mut c)).unwrap();
    assert_eq!(c, vec![3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn batched_gemm_matches_loop() {
    let q = Queue::new(0).unwrap();
    let (batch, m, n, kk) = (4usize, 3usize, 2usize, 5usize);
    let mut a = vec![0f32; batch * m * kk];
    let mut b = vec![0f32; batch * kk * n];
    for (i, x) in a.iter_mut().enumerate() {
        *x = (i % 7) as f32 - 3.0;
    }
    for (i, x) in b.iter_mut().enumerate() {
        *x = (i % 5) as f32 - 2.0;
    }
    let da = DeviceBuffer::alloc(&q, a.len() * 4).unwrap();
    let db = DeviceBuffer::alloc(&q, b.len() * 4).unwrap();
    let dc = DeviceBuffer::alloc(&q, batch * m * n * 4).unwrap();
    da.copy_from_host(bytes_of(&a)).unwrap();
    db.copy_from_host(bytes_of(&b)).unwrap();
    gemm(
        &q,
        SyclDType::F32,
        false,
        false,
        m as i64,
        n as i64,
        kk as i64,
        1.0,
        0.0,
        &da,
        &db,
        &dc,
        batch as i64,
        (m * kk) as i64,
        (kk * n) as i64,
        (m * n) as i64,
        0,
        0,
    )
    .unwrap();
    let mut c = vec![0f32; batch * m * n];
    dc.copy_to_host(bytes_of_mut(&mut c)).unwrap();
    // reference
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..kk {
                    acc += a[bi * m * kk + i * kk + p] * b[bi * kk * n + p * n + j];
                }
                let got = c[bi * m * n + i * n + j];
                assert!(
                    (got - acc).abs() < 1e-3,
                    "b{bi} [{i},{j}]: got {got} want {acc}"
                );
            }
        }
    }
}

#[test]
fn alloc_cost() {
    let q = Queue::new(0).unwrap();
    let sz = 4096 * 4096 * 4;
    // warmup
    for _ in 0..3 {
        let _b = DeviceBuffer::alloc(&q, sz).unwrap();
    }
    let t = std::time::Instant::now();
    for _ in 0..50 {
        let _b = DeviceBuffer::alloc(&q, sz).unwrap();
    }
    let big_ms = t.elapsed().as_secs_f64() * 1e3 / 50.0;
    println!("malloc_device+free (64MB): {big_ms:.3} ms/iter");

    let t = std::time::Instant::now();
    for _ in 0..50 {
        let _b = DeviceBuffer::alloc(&q, 16 * 1024).unwrap();
    }
    let small_ms = t.elapsed().as_secs_f64() * 1e3 / 50.0;
    println!("malloc_device+free (16KB): {small_ms:.3} ms/iter");

    // Assertions (tobert/lfm2d#3 item 7): the prints above are information
    // only, so a broken allocator measured 0 ms or NaN and still passed.
    // Bounds are order-of-magnitude, deliberately loose across Intel iGPUs
    // and data-center cards and driver versions — they catch allocator
    // breakage (negative/NaN/hang-adjacent timings, pooled buffers that are
    // not real), not performance regressions. 500 ms per 64 MB alloc/free
    // and 100 ms per 16 KB one are three orders above any plausible
    // oneAPI measurement.
    assert!(
        big_ms.is_finite() && big_ms > 0.0 && big_ms < 500.0,
        "64MB alloc+free regressed to {big_ms} ms/iter"
    );
    assert!(
        small_ms.is_finite() && small_ms > 0.0 && small_ms < 100.0,
        "16KB alloc+free regressed to {small_ms} ms/iter"
    );
    // Pooled small buffers are cheaper than 64 MB ones — if the pool ever
    // hands out stubs or the sizes collapse, this ordering flips.
    assert!(
        small_ms < big_ms,
        "16KB alloc ({small_ms} ms) costlier than 64MB ({big_ms} ms)"
    );
    // Functional pin: an allocated buffer must round-trip bytes.
    let b = DeviceBuffer::alloc(&q, 16 * 1024).unwrap();
    let pat: Vec<u8> = (0..16 * 1024).map(|i| (i % 251) as u8).collect();
    b.copy_from_host(&pat).unwrap();
    let mut back = vec![0u8; 16 * 1024];
    b.copy_to_host(&mut back).unwrap();
    assert_eq!(pat, back, "a pooled buffer did not round-trip its bytes");
}

#[test]
fn mmvq_speed() {
    let q = Queue::new(0).unwrap();
    let (n, k, m) = (4096usize, 4096usize, 1usize);
    let nblk = k / 32;
    let wbytes = n * nblk * 18; // q4_0 block = 18 bytes
    let w = DeviceBuffer::alloc(&q, wbytes).unwrap();
    let act = DeviceBuffer::alloc(&q, m * k * 4).unwrap();
    let out = DeviceBuffer::alloc(&q, m * n * 4).unwrap();
    w.copy_from_host(&vec![1u8; wbytes]).unwrap();
    act.copy_from_host(bytes_of(&vec![1f32; m * k])).unwrap();
    for _ in 0..3 {
        mmvq(&q, GgmlDType::Q4_0, &w, &act, &out, n, k, m).unwrap();
    }
    let t = std::time::Instant::now();
    for _ in 0..100 {
        mmvq(&q, GgmlDType::Q4_0, &w, &act, &out, n, k, m).unwrap();
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / 100.0;
    let gbps = wbytes as f64 / (ms * 1e-3) / 1e9;
    println!("mmvq q4_0 4096x4096 m=1: {ms:.3} ms  ({gbps:.1} GB/s)");
    // Sanity (item 7): a real GPU reads >= ~1 GB/s on a 94 MB weight scan
    // and no card does this in under a millisecond; ms/throughput measured
    // only, a dead kernel (0 weight bytes touched) printed 0 GB/s and
    // passed. Loose order-of-magnitude bounds, cross-driver tolerant.
    assert!(ms.is_finite() && ms > 0.0, "mmvq timed out at {ms} ms");
    assert!(
        gbps > 0.1 && gbps < 20_000.0,
        "implausible q4_0 mmvq throughput: {gbps} GB/s"
    );
    // Functional pin: q4_0 bytes of 0x11 are an fp16 scale of ~9.09 with
    // every nibble 1, so each of the 32 elements of a block dequantizes to
    // the same nonzero value and every act=1 block dot is that same
    // nonzero sum — an all-zero output means the kernel never wrote it.
    let mut outv = vec![0f32; n];
    out.copy_to_host(bytes_of_mut(&mut outv)).unwrap();
    assert!(
        outv.iter().any(|&v| v != 0.0),
        "mmvq wrote an all-zero output — the kernel is a no-op"
    );
}

#[test]
fn kernel_launch_overhead() {
    let q = Queue::new(0).unwrap();
    let n = 1024usize;
    let a = DeviceBuffer::alloc(&q, n * 4).unwrap();
    let b = DeviceBuffer::alloc(&q, n * 4).unwrap();
    a.copy_from_host(bytes_of(&vec![1f32; n])).unwrap();
    // warmup
    for _ in 0..20 {
        affine(&q, SyclDType::F32, &Layout::dense(), &a, &b, n, 2.0, 1.0).unwrap();
    }
    q.synchronize().unwrap();
    let t = std::time::Instant::now();
    let iters = 2000;
    for _ in 0..iters {
        affine(&q, SyclDType::F32, &Layout::dense(), &a, &b, n, 2.0, 1.0).unwrap();
    }
    q.synchronize().unwrap();
    let us_enqueue = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("affine(1024) enqueue: {us_enqueue:.1} us/iter (no per-op sync)");

    // with an alloc each iter (simulating the op dispatch)
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let out = DeviceBuffer::alloc(&q, n * 4).unwrap();
        affine(&q, SyclDType::F32, &Layout::dense(), &a, &out, n, 2.0, 1.0).unwrap();
    }
    q.synchronize().unwrap();
    let us_alloc = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("affine(1024) + pooled alloc: {us_alloc:.1} us/iter");

    // with a sync each iter (simulating the OLD behaviour)
    let t = std::time::Instant::now();
    for _ in 0..500 {
        affine(&q, SyclDType::F32, &Layout::dense(), &a, &b, n, 2.0, 1.0).unwrap();
        q.synchronize().unwrap();
    }
    let us_sync = t.elapsed().as_secs_f64() * 1e6 / 500.0;
    println!("affine(1024) + sync each: {us_sync:.1} us/iter");
    // Sanity (item 7): these prints were the only "check" — a kernel that
    // silently stopped executing still timed the empty queue and printed.
    // Enqueue overhead per launch is single-digit-to-low-hundreds of us on
    // oneAPI; 100 ms/iter is pathological (dead queue, hang-adjacent
    // driver), so the ceiling catches breakage, not drift. Per-iteration
    // ordering isn't pinned (device-side concurrency), only finiteness
    // and the absurd bound.
    for (name, us) in [
        ("enqueue", us_enqueue),
        ("+alloc", us_alloc),
        ("+sync", us_sync),
    ] {
        assert!(
            us.is_finite() && us > 0.0 && us < 100_000.0,
            "affine(1024) {name}: {us} us/iter is not a plausible launch cost"
        );
    }
}

#[test]
fn tiny_gemm_overhead() {
    let q = Queue::new(0).unwrap();
    // decode attention shapes
    let da = DeviceBuffer::alloc(&q, 128 * 4).unwrap();
    let db = DeviceBuffer::alloc(&q, 128 * 64 * 4).unwrap();
    let dc = DeviceBuffer::alloc(&q, 64 * 4).unwrap();
    da.copy_from_host(bytes_of(&vec![1f32; 128])).unwrap();
    db.copy_from_host(bytes_of(&vec![1f32; 128 * 64])).unwrap();
    for _ in 0..5 {
        gemm(
            &q,
            SyclDType::F32,
            false,
            false,
            1,
            64,
            128,
            1.0,
            0.0,
            &da,
            &db,
            &dc,
            1,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap();
    }
    q.synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..1000 {
        gemm(
            &q,
            SyclDType::F32,
            false,
            false,
            1,
            64,
            128,
            1.0,
            0.0,
            &da,
            &db,
            &dc,
            1,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap();
    }
    let enq = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    q.synchronize().unwrap();
    let tot = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    println!("tiny gemm(1x64x128): enqueue {enq:.1}us  total {tot:.1}us");
    // Sanity (item 7): the prints alone could not fail — an oneMKL call
    // that silently stopped computing timed just as fast. The gemm is
    // ones(1x128) @ ones(128x64) with alpha=1, beta=0, so every output is
    // the exact fp32 sum of 128 ones; pin the values, then the timings.
    let mut dc_v = vec![0f32; 64];
    dc.copy_to_host(bytes_of_mut(&mut dc_v)).unwrap();
    for (i, &v) in dc_v.iter().enumerate() {
        assert!(
            (v - 128.0).abs() < 1e-3,
            "gemm row {i}: {v}, expected 128 — oneMKL gemm broke"
        );
    }
    for (name, us) in [("enqueue", enq), ("total", tot)] {
        assert!(
            us.is_finite() && us > 0.0 && us < 100_000.0,
            "tiny gemm {name}: {us} us/iter is not a plausible cost"
        );
    }
    assert!(enq <= tot, "gemm enqueue {enq}us exceeded total {tot}us");

    // mmvq decode shape (qwen 0.6b q_proj ~ 2048x1024 q4k)
    let nblk = 1024 / 256;
    let w = DeviceBuffer::alloc(&q, 2048 * nblk * 144).unwrap();
    let act = DeviceBuffer::alloc(&q, 1024 * 4).unwrap();
    let o = DeviceBuffer::alloc(&q, 2048 * 4).unwrap();
    // Functional probe first: q4_k bytes of 0x11 give nonzero (tiny) f16
    // block scales and nibbles, so the dequantized dot product cannot be
    // exactly zero for all 2048 rows — an all-zero output means the kernel
    // never wrote it. (The timing loop below then zeroes the weights as
    // before; mmvq's cost is data-independent.)
    w.copy_from_host(&vec![0x11u8; 2048 * nblk * 144]).unwrap();
    act.copy_from_host(bytes_of(&vec![1f32; 1024])).unwrap();
    mmvq(&q, GgmlDType::Q4K, &w, &act, &o, 2048, 1024, 1).unwrap();
    q.synchronize().unwrap();
    let mut o_v = vec![0f32; 2048];
    o.copy_to_host(bytes_of_mut(&mut o_v)).unwrap();
    assert!(
        o_v.iter().any(|&v| v != 0.0),
        "mmvq q4_k wrote an all-zero output — the kernel is a no-op"
    );
    w.copy_from_host(&vec![0u8; 2048 * nblk * 144]).unwrap();
    for _ in 0..5 {
        mmvq(&q, GgmlDType::Q4K, &w, &act, &o, 2048, 1024, 1).unwrap();
    }
    q.synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..1000 {
        mmvq(&q, GgmlDType::Q4K, &w, &act, &o, 2048, 1024, 1).unwrap();
    }
    let enq = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    q.synchronize().unwrap();
    let tot = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    println!("mmvq q4k 2048x1024 m=1: enqueue {enq:.1}us  total {tot:.1}us");
    for (name, us) in [("enqueue", enq), ("total", tot)] {
        assert!(
            us.is_finite() && us > 0.0 && us < 100_000.0,
            "mmvq q4k {name}: {us} us/iter is not a plausible cost"
        );
    }
    assert!(enq <= tot, "mmvq enqueue {enq}us exceeded total {tot}us");
}
