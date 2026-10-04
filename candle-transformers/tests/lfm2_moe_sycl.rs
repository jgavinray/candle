use candle::quantized::gguf_file;
use candle::{Device, Result};
use candle_transformers::models::quantized_lfm2_moe::{Model, State};
fn device() -> Result<Device> {
    if std::env::var("LFM2_SYCL_TEST").is_ok() {
        #[cfg(feature = "sycl")]
        return Device::new_sycl(0);
    }
    Ok(Device::Cpu)
}
fn model() -> Result<Model> {
    let mut f = std::io::Cursor::new(include_bytes!("fixtures/lfm2-moe/tiny.gguf"));
    let ct = gguf_file::Content::read(&mut f)?;
    Model::from_gguf(ct, &mut f, &device()?)
}
fn output(model: &Model, tokens: &[u32], state: &mut State) -> Result<Vec<f32>> {
    model
        .forward(tokens, state)?
        .flatten_all()?
        .to_vec1::<f32>()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < 2e-5, "logit {i}: {a} vs {b}");
    }
}
#[test]
fn whole_and_incremental_logits_match_numpy() -> Result<()> {
    let m = model()?;
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lfm2-moe/reference.json")).unwrap();
    for c in cases.as_array().unwrap() {
        let tokens: Vec<u32> = serde_json::from_value(c["tokens"].clone()).unwrap();
        let logits: Vec<Vec<f32>> = serde_json::from_value(c["logits"].clone()).unwrap();
        close(
            &output(&m, &tokens, &mut m.new_state())?,
            logits.last().unwrap(),
        );
        let mut state = m.new_state();
        for (i, &t) in tokens.iter().enumerate() {
            close(&output(&m, &[t], &mut state)?, &logits[i]);
        }
        for split in 1..tokens.len() {
            let mut state = m.new_state();
            output(&m, &tokens[..split], &mut state)?;
            close(
                &output(&m, &tokens[split..], &mut state)?,
                logits.last().unwrap(),
            );
        }
    }
    Ok(())
}
#[test]
fn snapshot_a_b_a_and_rejections_leave_prefix_usable() -> Result<()> {
    let m = model()?;
    let mut prefix = m.new_state();
    output(&m, &[1, 2, 3], &mut prefix)?;
    let mut branch = prefix.clone();
    let a = output(&m, &[4, 5, 6], &mut branch)?;
    assert_eq!(prefix.len(), 3);
    output(&m, &[9, 8, 7], &mut prefix.clone())?;
    close(&a, &output(&m, &[4, 5, 6], &mut prefix.clone())?);
    close(&a, &output(&m, &[1, 2, 3, 4, 5, 6], &mut m.new_state())?);
    for invalid in [vec![], vec![16], vec![1; 32]] {
        assert!(m.forward(&invalid, &mut prefix).is_err());
        assert_eq!(prefix.len(), 3);
    }
    assert!(model()?.forward(&[4], &mut prefix).is_err());
    close(&a, &output(&m, &[4, 5, 6], &mut prefix)?);
    Ok(())
}

/// Same as `close` but at a caller-chosen bound. Honest scope note
/// (tobert/lfm2d#3 item 7 round 2): with this F32-stored fixture both the
/// batched and the token-by-token call run the SAME dense path (CPU
/// `index_select` + f32 matmul; on SYCL the gather + strided-batch f32 GEMM
/// fallback), so "different kernel shapes" is NOT what this tolerance
/// absorbs — only GEMM blocking/accumulation order at 64 tasks vs 2, which
/// stays far below 1e-3. Its real job is catching gross routing or dispatch
/// regressions (wrong expert picked, threshold off-by-one, cache staleness)
/// at the largest single call the fixture allows. Genuine kernel-shape
/// divergence (integer mat-vec vs grouped dequant+f16 GEMM) awaits a
/// quantized (Q4K/Q5K/Q6_K) twin of tiny.gguf regenerated through
/// reference.py.
fn close_loose(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < tol, "logit {i}: {a} vs {b}");
    }
}

/// 32 tokens x top-2 = 64 routed tasks — the largest single call the
/// fixture allows (context_length = 32, and `forward` rejects anything
/// longer) and exactly `GROUPED_MIN_TASKS` (`quantized/sycl.rs`), the
/// grouped-prefill dispatch threshold. What actually runs, per backend
/// (tobert/lfm2d#3 item 7 round 2 — the PR body's source of truth):
///  * CPU (the fast arm here): the fixture's experts are stored F32 and
///    `Experts::new` dequantizes to a dense CPU tensor on CPU devices, so
///    `quantized_lfm2_moe`'s `Cpu` reference arm runs and
///    `indexed_moe_forward` is never called — the SYCL threshold is not
///    even evaluated. This pins model-level batch consistency at the
///    threshold batch: routing, softmax weights, and per-layer accumulation
///    over a 64-task prefill must agree with the incremental walk.
///  * B70/oneAPI: the 64-task count REACHES the grouped-dispatch CHECK,
///    but the same F32 dtype gate (`sycl.rs`: grouped requires
///    `Q4K | Q5K | Q6K`) sends this fixture to the dense gather +
///    strided-batch-GEMM fallback — the threshold is evaluated, the
///    grouped BRANCH is never taken. True grouped-kernel coverage needs a
///    quantized (Q4K/Q5K/Q6_K) twin of tiny.gguf regenerated through
///    reference.py — a fixture change for whoever provisions B70, not a
///    binary to fabricate here.
/// Fixture limits (reference.py): 3 experts, expert_used_count 2, vocab
/// 16, context 32 cap the single call at 64 tasks.
#[test]
fn batch_32_prefill_matches_token_by_token() -> Result<()> {
    let m = model()?;
    let tokens: Vec<u32> = (0..32).map(|i| (i % 16) as u32).collect();
    let whole = output(&m, &tokens, &mut m.new_state())?;
    let mut state = m.new_state();
    let mut step = Vec::new();
    for &t in &tokens {
        step = output(&m, &[t], &mut state)?;
    }
    // The CPU arm keeps this cheap (one extra 32-token prefill) while
    // pinning model-level batch consistency; the SYCL arm is what makes
    // it a real kernel-path test.
    close_loose(&whole, &step, 1e-3);
    Ok(())
}
