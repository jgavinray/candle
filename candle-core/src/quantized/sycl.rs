//! GGUF quantized storage for the SYCL backend: `load_quantized`, GPU
//! `dequantize` for all 12 GGUF quant types, `quantize`/`quantize_onto`
//! (candle's CPU quantizer plus an upload), `embedding`, and `QMatMul::fwd`,
//! which dispatches between a fused mat-vec that keeps the weight quantized
//! (small `m`, the decode path) and dequantize plus a oneMKL GEMM (prefill).
#![allow(unused)]
use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::sycl_backend::{k, storage_from_buffer, storage_view, SyclError};
use crate::{DType, Layout, Result, Shape, SyclDevice, SyclStorage};

/// Largest `m` (activation rows) still served by the integer mat-vec kernel
/// rather than dequantize-the-weight plus a GEMM. Below it the mat-vec wins;
/// above it the GEMM amortizes materializing the whole dense weight.
const MMVQ_MAX_M: usize = 32;

/// The dense dtype of an unquantized GGUF weight, which has no blocks to decode.
fn dense_dtype(dt: GgmlDType) -> Option<DType> {
    match dt {
        GgmlDType::F32 => Some(DType::F32),
        GgmlDType::F16 => Some(DType::F16),
        GgmlDType::BF16 => Some(DType::BF16),
        _ => None,
    }
}

fn nyi<T>(what: &str) -> Result<T> {
    Err(crate::Error::Sycl(
        SyclError::msg(format!("quantized/sycl: {what} not implemented yet")).into(),
    ))
}

fn to_k_dtype(dt: GgmlDType) -> k::GgmlDType {
    use k::GgmlDType as K;
    match dt {
        GgmlDType::F32 => K::F32,
        GgmlDType::F16 => K::F16,
        GgmlDType::BF16 => K::BF16,
        GgmlDType::Q4_0 => K::Q4_0,
        GgmlDType::Q4_1 => K::Q4_1,
        GgmlDType::Q5_0 => K::Q5_0,
        GgmlDType::Q5_1 => K::Q5_1,
        GgmlDType::Q8_0 => K::Q8_0,
        GgmlDType::Q8_1 => K::Q8_1,
        GgmlDType::Q2K => K::Q2K,
        GgmlDType::Q3K => K::Q3K,
        GgmlDType::Q4K => K::Q4K,
        GgmlDType::Q5K => K::Q5K,
        GgmlDType::Q6K => K::Q6K,
        GgmlDType::Q8K => K::Q8K,
    }
}

pub struct QSyclStorage {
    data: k::DeviceBuffer,
    dtype: GgmlDType,
    elem_count: usize,
    device: SyclDevice,
    /// One-entry cache for the dequantized weight. `data` is never mutated
    /// after construction, so a cached dequant of `elem_count` values stays
    /// valid for the storage's lifetime; the entry is dropped if a different
    /// `elem_count` is requested (the only variation callers use).
    dequant_cache: parking_lot::Mutex<Option<(usize, bool, SyclStorage)>>,
}

impl QSyclStorage {
    fn from_cpu_quant(
        device: &SyclDevice,
        dtype: GgmlDType,
        cpu: &dyn super::QuantizedType,
        elem_count: usize,
    ) -> Result<Self> {
        let bytes = cpu.storage_size_in_bytes();
        let host = unsafe { std::slice::from_raw_parts(cpu.as_ptr(), bytes) };
        let data = device.alloc_bytes(bytes)?;
        data.copy_from_host(host)
            .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
        Ok(Self {
            data,
            dtype,
            elem_count,
            device: device.clone(),
            dequant_cache: parking_lot::Mutex::new(None),
        })
    }
    /// Host-side orchestration for the grouped MoE prefill: dequantize
    /// the whole expert stack to f16 once (cached), gather the task
    /// rows, run one f16 GEMM per expert over its compacted rows, and
    /// scatter the results back to task order. Uses only existing
    /// kernels (dequantize_f16, index_select, cast, oneMKL gemm).
    #[allow(clippy::too_many_arguments)]
    fn indexed_moe_grouped(
        &self,
        act: &SyclStorage,
        ids: &SyclStorage,
        batch: usize,
        topk: usize,
        input_dim1: usize,
        num_experts: usize,
        n: usize,
        k: usize,
    ) -> Result<(SyclStorage, Shape)> {
        let werr = |e: k::SyclError| crate::Error::Sycl(SyclError::msg(e.to_string()).into());
        let tasks = batch * topk;
        // ids to host: tasks * 4 bytes, a few KiB.
        let mut id_host = vec![0u8; tasks * 4];
        ids.buf()
            .copy_to_host(&mut id_host)
            .map_err(werr)?;
        let expert_of_task: Vec<u32> = id_host
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for &e in &expert_of_task {
            if e as usize >= num_experts {
                crate::bail!("indexed_moe_grouped: expert id {e} out of range for {num_experts} experts");
            }
        }
        // Compact position of each task's result row, and the gather
        // index (compact row -> activation row).
        let mut counts = vec![0u32; num_experts];
        for &e in &expert_of_task {
            counts[e as usize] += 1;
        }
        let mut off = vec![0u32; num_experts];
        let mut acc = 0u32;
        for (e, c) in counts.iter().enumerate() {
            off[e] = acc;
            acc += c;
        }
        let mut cursor = off.clone();
        let mut pos_of_task = vec![0u32; tasks];
        let mut gather = vec![0u32; tasks];
        // Source rows: input_dim1 == 1 -> act is (batch, k), task t reads
        // batch row t/topk; input_dim1 == topk -> act is (tasks, k), task
        // t reads row t. (The dispatch guard ensures one of the two.)
        let src_rows = batch * input_dim1;
        for (t, &e) in expert_of_task.iter().enumerate() {
            let p = cursor[e as usize];
            cursor[e as usize] = p + 1;
            pos_of_task[t] = p;
            gather[p as usize] = if input_dim1 == 1 { (t / topk) as u32 } else { t as u32 };
        }
        debug_assert!(gather.iter().max().copied().unwrap_or(0) < src_rows as u32);
        // f16 activation for the GEMMs, gathered straight from the
        // source rows — no (batch*topk) expansion materialized.
        let act_f16 = if act.dtype() == DType::F16 {
            storage_view(act)
        } else {
            act.to_dtype_raw(&Layout::contiguous(src_rows * k), DType::F16)?
        };
        let to_bytes = |v: &[u32]| -> Vec<u8> {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        };
        // gather compact rows: index_select over dim 0.
        let gather_buf = self.device.alloc_bytes(tasks * 4)?;
        gather_buf.copy_from_host(&to_bytes(&gather)).map_err(werr)?;
        let gather_ids = storage_from_buffer(&self.device, gather_buf, DType::U32, tasks);
        let compact = act_f16.index_select_raw(
            &gather_ids,
            &Layout::contiguous((tasks, k)),
            &Layout::contiguous(tasks),
            0,
        )?;
        // Dequantize the whole expert stack to f16 (one kernel over all
        // blocks; cached by `dequantize_f16`).
        let wf16 = self.dequantize_f16(num_experts * n * k)?;
        // Per-expert GEMM into a compact result buffer.
        let tmp = self.device.alloc_bytes(tasks * n * 2)?;
        for e in 0..num_experts {
            let cnt = counts[e] as usize;
            if cnt == 0 {
                continue;
            }
            let o = off[e] as usize;
            let lhs_l = Layout::new((cnt, k).into(), vec![k, 1], o * k);
            // Weight rows for expert e start at e*n*k elements; the
            // (k, n) stride-(1, k) view orients the (n, k) stack for
            // the GEMM (same trick as `fwd`'s `gemm` closure).
            let rhs_l = Layout::new((k, n).into(), vec![1, k], e * n * k);
            let res = compact.matmul_raw(&wf16, (1, cnt, n, k), &lhs_l, &rhs_l)?;
            // SAFETY: the offset is within the buffer allocated above;
            // both views stay alive for the copy.
            unsafe {
                tmp.view_at(o * n * 2)
                    .copy_from_device(res.buf(), cnt * n * 2)
                    .map_err(werr)?;
            }
        }
        // Scatter compact rows back to task order, then widen to f32 to
        // match the mat-vec path's output dtype.
        let scatter_buf = self.device.alloc_bytes(tasks * 4)?;
        scatter_buf
            .copy_from_host(&to_bytes(&pos_of_task))
            .map_err(werr)?;
        let scatter_ids = storage_from_buffer(&self.device, scatter_buf, DType::U32, tasks);
        let tmp_storage = storage_from_buffer(&self.device, tmp, DType::F16, tasks * n);
        let out_f16 = tmp_storage.index_select_raw(
            &scatter_ids,
            &Layout::contiguous((tasks, n)),
            &Layout::contiguous(tasks),
            0,
        )?;
        let out = out_f16.to_dtype_raw(&Layout::contiguous(tasks * n), DType::F32)?;
        Ok((out, Shape::from((batch, topk, n))))
    }
}

impl QSyclStorage {
    pub fn zeros(device: &SyclDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let n_blocks = elem_count / dtype.block_size();
        let bytes = n_blocks * dtype.type_size();
        let data = device.alloc_bytes(bytes)?;
        data.memset_zero()
            .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
        Ok(Self {
            data,
            dtype,
            elem_count,
            device: device.clone(),
            dequant_cache: parking_lot::Mutex::new(None),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &SyclDevice {
        &self.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len_bytes()
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len_bytes()];
        self.data
            .copy_to_host(&mut out)
            .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.as_ptr() as *const u8)
    }

    /// Dequantize `elem_count` values into an f32 `SyclStorage`. The result
    /// is cached per storage (the weight is immutable): repeated prefills of
    /// the same layer skip the dequantize + write entirely. Callers get a
    /// zero-copy alias; the cache entry owns the allocation.
    pub fn dequantize(&self, elem_count: usize) -> Result<SyclStorage> {
        self.cached_dequant(elem_count, false)
    }

    /// Dequantize into an f16 `SyclStorage`, cached like [`Self::dequantize`].
    pub fn dequantize_f16(&self, elem_count: usize) -> Result<SyclStorage> {
        self.cached_dequant(elem_count, true)
    }

    fn cached_dequant(&self, elem_count: usize, f16_out: bool) -> Result<SyclStorage> {
        {
            let cache = self.dequant_cache.lock();
            if let Some((n, f16, owner)) = cache.as_ref() {
                if *n == elem_count && *f16 == f16_out {
                    return Ok(storage_view(owner));
                }
            }
        }
        let out = self.dequantize_uncached(elem_count, f16_out)?;
        let mut cache = self.dequant_cache.lock();
        match cache.as_ref() {
            Some((n, f16, _)) if *n == elem_count && *f16 == f16_out => {}
            _ => *cache = Some((elem_count, f16_out, out)),
        }
        drop(cache);
        let cache = self.dequant_cache.lock();
        Ok(storage_view(&cache.as_ref().unwrap().2))
    }

    fn dequantize_uncached(&self, elem_count: usize, f16_out: bool) -> Result<SyclStorage> {
        if f16_out {
            let blk = self.dtype.block_size();
            if blk > 1 && elem_count.is_multiple_of(blk) {
                let out = self.device.new_storage(DType::F16, elem_count)?;
                k::dequantize_f16(
                    self.device.q(),
                    to_k_dtype(self.dtype),
                    &self.data,
                    out.buf(),
                    elem_count / blk,
                )
                .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
                return Ok(out);
            }
            // F32/F16/BF16 (block_size 1) have no dequantize kernel; reinterpret.
            let f32s = self.dequantize_uncached(elem_count, false)?;
            return f32s.to_dtype_raw(&Layout::contiguous(elem_count), DType::F16);
        }
        self.dequantize_f32(elem_count)
    }

    fn dequantize_f32(&self, elem_count: usize) -> Result<SyclStorage> {
        let out = self.device.new_storage(DType::F32, elem_count)?;
        let w = |e: k::SyclError| crate::Error::Sycl(SyclError::msg(e.to_string()).into());
        match self.dtype {
            GgmlDType::F32 => {
                out.buf()
                    .copy_from_device(&self.data, elem_count * 4)
                    .map_err(w)?;
            }
            GgmlDType::F16 | GgmlDType::BF16 => {
                // reinterpret the raw halves as a SyclStorage, then cast to f32.
                let half = storage_from_buffer(
                    &self.device,
                    self.raw_clone()?,
                    if self.dtype == GgmlDType::F16 {
                        DType::F16
                    } else {
                        DType::BF16
                    },
                    elem_count,
                );
                let f32s = half.to_dtype_raw(&Layout::contiguous(elem_count), DType::F32)?;
                out.buf()
                    .copy_from_device(f32s.buf(), elem_count * 4)
                    .map_err(w)?;
            }
            other => {
                let n_blocks = elem_count / other.block_size();
                k::dequantize(
                    self.device.q(),
                    to_k_dtype(other),
                    &self.data,
                    out.buf(),
                    n_blocks,
                )
                .map_err(w)?;
            }
        }
        Ok(out)
    }

    fn raw_clone(&self) -> Result<k::DeviceBuffer> {
        let b = self.device.alloc_bytes(self.data.len_bytes())?;
        b.copy_from_device(&self.data, self.data.len_bytes())
            .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
        Ok(b)
    }

    /// `QMatMul` forward. `self_shape = (n, k)` (weight is stored transposed);
    /// `storage`/`layout` is the activation with shape `(.., k)`, result `(.., n)`.
    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &SyclStorage,
        layout: &Layout,
    ) -> Result<(SyclStorage, Shape)> {
        let (n, kk) = self_shape.dims2()?;
        let src_dims = layout.shape().dims().to_vec();
        let m: usize = src_dims[..src_dims.len() - 1].iter().product();
        if src_dims[src_dims.len() - 1] != kk {
            crate::bail!("qmatmul: input {layout:?} incompatible with weight {self_shape:?}");
        }
        let mut out_dims = src_dims.clone();
        *out_dims.last_mut().unwrap() = n;
        let out_shape = Shape::from(out_dims);
        let werr = |e: k::SyclError| crate::Error::Sycl(SyclError::msg(e.to_string()).into());
        let kdt = to_k_dtype(self.dtype);
        let dense = layout.start_offset() == 0 && layout.is_contiguous();

        // The weight is stored (n, k); a transposed view makes `mm_operand` see
        // `transb`, so no copy is needed to orient it.
        let gemm = |act: &SyclStorage, w: &SyclStorage| {
            let w_t = Layout::new((kk, n).into(), vec![1, kk], 0);
            act.matmul_raw(w, (1, m, n, kk), &Layout::contiguous((m, kk)), &w_t)
        };
        let as_act_dtype = |out: SyclStorage| {
            if out.dtype() == storage.dtype() {
                Ok(out)
            } else {
                out.to_dtype_raw(&Layout::contiguous((m, n)), storage.dtype())
            }
        };

        // An unquantized weight has nothing to decode: a plain GEMM against the
        // raw buffer, in the weight's own dtype. Handling it here is what lets
        // `QMatMul::from_arc` leave it alone instead of widening it to F32.
        if let Some(wdt) = dense_dtype(self.dtype) {
            // Borrowed: `self.data` stays the owner for the call's duration.
            let w = storage_from_buffer(&self.device, unsafe { self.data.view() }, wdt, n * kk);
            let act_owned;
            let act: &SyclStorage = if storage.dtype() == wdt && dense {
                storage
            } else {
                act_owned = storage.to_dtype_raw(layout, wdt)?;
                &act_owned
            };
            return Ok((as_act_dtype(gemm(act, &w)?)?, out_shape));
        }

        let blk = self.dtype.block_size();
        let int_ok = k::mmvq_q8_block(kdt).is_some_and(|b| kk.is_multiple_of(b));
        let takes_mmvq =
            (m <= MMVQ_MAX_M && int_ok) || (m <= 8 && blk != 1 && kk.is_multiple_of(blk));

        // Half-precision pipeline: stay in f16 throughout. The mat-vec kernel
        // reads and writes f16 itself, saving the casts around every quantized
        // linear, and an f16 GEMM can use the matrix engines that an f32 one
        // cannot.
        if dense && storage.dtype() == DType::F16 {
            if m <= MMVQ_MAX_M && int_ok {
                let out = self.device.new_storage(DType::F16, m * n)?;
                k::mmvq_q8(
                    self.device.q(),
                    kdt,
                    &self.data,
                    storage.buf(),
                    true,
                    out.buf(),
                    true,
                    n,
                    kk,
                    m,
                )
                .map_err(werr)?;
                return Ok((out, out_shape));
            }
            if !takes_mmvq {
                return Ok((gemm(storage, &self.dequantize_f16(n * kk)?)?, out_shape));
            }
        }

        // Otherwise work in f32: a dense (m, k) activation, cast if it is not
        // already one.
        let act_owned;
        let act: &SyclStorage = if storage.dtype() == DType::F32 && dense {
            storage
        } else {
            act_owned = storage.to_dtype_raw(layout, DType::F32)?;
            &act_owned
        };
        let out = if takes_mmvq {
            // Decode: fused mat-vec, the weight stays quantized in memory. The
            // integer kernel where one exists, float dequant-and-dot otherwise.
            let out = self.device.new_storage(DType::F32, m * n)?;
            let (q, w, a, o) = (self.device.q(), &self.data, act.buf(), out.buf());
            if !k::mmvq_q8(q, kdt, w, a, false, o, false, n, kk, m).map_err(werr)? {
                k::mmvq(q, kdt, w, a, o, n, kk, m).map_err(werr)?;
            }
            out
        } else {
            // Prefill / fallback: dequantize the whole weight, one oneMKL GEMM.
            gemm(act, &self.dequantize(n * kk)?)?
        };
        Ok((as_act_dtype(out)?, out_shape))
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &SyclStorage,
        ids_l: &Layout,
    ) -> Result<SyclStorage> {
        let blk = self.dtype.block_size();
        // Dequantize only the gathered rows, rather than the whole table. A
        // non-zero `start_offset` is fine here, it only moves the base pointer.
        if dense_dtype(self.dtype).is_none()
            && ids.dtype() == DType::U32
            && ids_l.is_contiguous()
            && hidden.is_multiple_of(blk)
        {
            let n_ids = ids_l.shape().elem_count();
            let out = self.device.new_storage(DType::F32, n_ids * hidden)?;
            let ids_buf = unsafe {
                ids.buf()
                    .view_at(ids_l.start_offset() * DType::U32.size_in_bytes())
            };
            k::get_rows(
                self.device.q(),
                to_k_dtype(self.dtype),
                &self.data,
                &ids_buf,
                out.buf(),
                n_ids,
                hidden / blk,
            )
            .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
            return Ok(out);
        }
        let w = self.dequantize(rows * hidden)?;
        w.index_select_raw(ids, &Layout::contiguous((rows, hidden)), ids_l, 0)
    }

    pub fn quantize(&mut self, src: &SyclStorage) -> Result<()> {
        // No on-device quantizer yet: pull to host, use candle's CPU quantizer,
        // upload the blocks. Correct, and quantization is a one-off at load time.
        let cpu = src.to_cpu_storage()?;
        self.quantize_onto(&cpu)
    }
    pub fn quantize_imatrix(&mut self, _src: &SyclStorage, _w: &[f32], _n: usize) -> Result<()> {
        nyi("quantize_imatrix")
    }
    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _w: &[f32],
        _n: usize,
    ) -> Result<()> {
        nyi("quantize_imatrix_onto")
    }
    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        let f32s = src.as_slice::<f32>()?;
        let mut cpu = self.dtype.cpu_zeros(f32s.len());
        cpu.from_float(f32s);
        *self = QSyclStorage::from_cpu_quant(&self.device, self.dtype, cpu.as_ref(), f32s.len())?;
        Ok(())
    }

    pub fn fwd_via_dequant(&self) -> Result<()> {
        Ok(())
    }

    /// Indexed MoE mat-vec (decode path of `quantized_lfm2_moe`). `self` is the
    /// `[num_experts, n, k]` weight stack; `input` is a dense `(batch,
    /// topk or 1, k)` f32 (or f16) activation and `ids` a dense `(batch, topk)`
    /// u32 expert-id buffer, both on this device. Returns `(batch, topk, n)`.
    /// The integer kernel is used where one exists (`mmvq_q8_block`); other
    /// dtypes error rather than silently falling back, matching the CUDA path.
    pub fn indexed_moe_forward(
        &self,
        self_shape: &Shape,
        input: &SyclStorage,
        input_l: &Layout,
        ids: &SyclStorage,
        ids_l: &Layout,
    ) -> Result<(SyclStorage, Shape)> {
        let (num_experts, n, k) = self_shape.dims3()?;
        let src_dims = input_l.shape().dims().to_vec();
        // The activation is `(batch, k)` with one row broadcast across the
        // task's topk experts, or `(batch, topk, k)` with a row per task —
        // the same two shapes the CUDA kernels accept (`input_dim1`).
        let (batch, input_dim1) = match src_dims.len() {
            2 => {
                if src_dims[1] != k {
                    crate::bail!("indexed_moe_forward: input last dim {} != weight k {k}", src_dims[1]);
                }
                (src_dims[0], 1)
            }
            3 => {
                if src_dims[2] != k {
                    crate::bail!("indexed_moe_forward: input last dim {} != weight k {k}", src_dims[2]);
                }
                (src_dims[0], src_dims[1])
            }
            _ => crate::bail!("indexed_moe_forward: input must be rank 2 or 3, got {src_dims:?}"),
        };
        let topk = ids_l.shape().dims().last().copied().unwrap_or(0);
        let ids_rows = ids_l.shape().elem_count();
        if ids_rows != batch * topk {
            crate::bail!(
                "indexed_moe_forward: ids count {ids_rows} does not match batch {batch} x topk {topk}"
            );
        }
        if input_dim1 != 1 && input_dim1 != topk {
            crate::bail!(
                "indexed_moe_forward: input_dim1 {input_dim1} must be 1 or topk {topk}"
            );
        }
        if !input_l.is_contiguous() || input_l.start_offset() != 0 {
            crate::bail!("indexed_moe_forward: input must be dense and contiguous");
        }
        if !ids_l.is_contiguous() || ids_l.start_offset() != 0 || ids.dtype() != DType::U32 {
            crate::bail!("indexed_moe_forward: ids must be dense contiguous u32 at offset 0");
        }
        // The integer kernel reads the activation as f32 or f16; a BF16
        // activation would be reinterpreted as f32. The CUDA path likewise
        // takes an f32 activation.
        if !matches!(input.dtype(), DType::F32 | DType::F16) {
            crate::bail!(
                "indexed_moe_forward: activation dtype {:?} is not f32 or f16",
                input.dtype()
            );
        }
        if num_experts == 0 || n == 0 || k == 0 || batch == 0 || topk == 0 {
            crate::bail!(
                "indexed_moe_forward: empty shape (experts {num_experts}, n {n}, k {k}, batch {batch}, topk {topk})"
            );
        }
        if !k.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "indexed_moe_forward: k {k} is not a multiple of block size {}",
                self.dtype.block_size()
            );
        }
        let werr = |e: k::SyclError| crate::Error::Sycl(SyclError::msg(e.to_string()).into());
        // Grouped prefill: at large task counts the integer mat-vec re-reads
        // every routed expert matrix from HBM per task row (~7 us/row
        // measured on the B70; the 402 MB expert stack defeats the L2).
        // Dequantizing the stack once and running per-expert GEMMs over the
        // gathered rows amortizes that read (dequant measured 6.24 ms for
        // this stack vs 29.8 ms for the mat-vec at batch 512). Dispatched
        // BEFORE the row expansion: the gather maps task -> source row
        // itself, and materializing batch*topk expanded rows first costs
        // 3.5 s of d2d copies at batch 512 (measured), dwarfing everything
        // else. The dequantized stack is cached by `dequantize_f16`, so
        // steady-state prefills skip it entirely.
        const GROUPED_MIN_TASKS: usize = 64;
        if batch * topk >= GROUPED_MIN_TASKS
            && (input_dim1 == 1 || input_dim1 == topk)
            && matches!(self.dtype, GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K)
        {
            return self.indexed_moe_grouped(input, ids, batch, topk, input_dim1, num_experts, n, k);
        }
        // The activation is one row per routed task. When input_dim1 == 1 the
        // CUDA path re-reads the batch row for every topk entry; quantizing
        // `batch * topk` rows would repeat work, so broadcast it instead — the
        // kernel indexes rows by task id, so materialize the expanded rows once.
        let act_owned;
        let act: &SyclStorage = if input_dim1 == 1 && topk > 1 {
            // The CUDA path re-reads the batch row for every topk entry; the
            // SYCL kernel indexes rows by task id, so materialize the expanded
            // rows once with device-to-device copies.
            let expanded = self.device.new_storage(input.dtype(), batch * topk * k)?;
            let row_bytes = k * input.dtype().size_in_bytes();
            for b in 0..batch {
                for t in 0..topk {
                    // SAFETY: offsets are within the (batch*topk*k)-element
                    // buffers just allocated; both views stay alive for the
                    // copy.
                    unsafe {
                        expanded
                            .buf()
                            .view_at((b * topk + t) * row_bytes)
                            .copy_from_device(&input.buf().view_at(b * row_bytes), row_bytes)
                            .map_err(werr)?;
                    }
                }
            }
            act_owned = expanded;
            &act_owned
        } else {
            input
        };
        let out = self.device.new_storage(DType::F32, batch * topk * n)?;
        let launched = k::indexed_moe_q8(
            self.device.q(),
            to_k_dtype(self.dtype),
            &self.data,
            act.buf(),
            act.dtype() == DType::F16,
            ids.buf(),
            out.buf(),
            false,
            n,
            k,
            batch,
            topk,
        )
        .map_err(werr)?;
        if launched {
            return Ok((out, Shape::from((batch, topk, n))));
        }

        // Dense stacks (F32/F16/BF16) have no blocks to dot. Gather each
        // task's expert matrix and run one strided-batch GEMM, computing the
        // same thing as `index_select(ids).matmul` on the CPU reference.
        //
        // `get_rows` is not usable here: it dequantizes through the block
        // dispatch, which has no case for the dense dtypes (block size 1).
        // Byte-copying the expert matrices is exact for them.
        if !matches!(self.dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) {
            crate::bail!(
                "indexed_moe_forward: no SYCL kernel for {:?}; the CPU path is the reference",
                self.dtype
            );
        }
        let mut id_host = vec![0u8; ids_rows * DType::U32.size_in_bytes()];
        ids.buf()
            .copy_to_host(&mut id_host)
            .map_err(werr)?;
        let expert_ids: Vec<u32> = id_host
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let dense = dense_dtype(self.dtype)
            .ok_or_else(|| crate::Error::Sycl(SyclError::msg("indexed_moe_forward: not a dense dtype").into()))?;
        let row_bytes = n * k * dense.size_in_bytes();
        let routed = self.device.new_storage(dense, ids_rows * n * k)?;
        for (ti, &expert) in expert_ids.iter().enumerate() {
            if expert as usize >= num_experts {
                crate::bail!(
                    "indexed_moe_forward: expert id {expert} out of range for {num_experts} experts"
                );
            }
            // SAFETY: the expert bound is checked above and both offsets are
            // within the buffers allocated here.
            unsafe {
                routed
                    .buf()
                    .view_at(ti * row_bytes)
                    .copy_from_device(
                        &self.data.view_at(expert as usize * row_bytes),
                        row_bytes,
                    )
                    .map_err(werr)?;
            }
        }
        // C[ti, 0..n] = act[ti, 0..k] . W_ti(n, k): A is one activation row
        // per task, B is that task's expert matrix (`transb`), batched.
        // oneMKL takes one element type for all three operands.
        let a = act.to_dtype_raw(&Layout::contiguous((ids_rows, k)), DType::F32)?;
        let b = routed.to_dtype_raw(&Layout::contiguous((ids_rows, n * k)), DType::F32)?;
        let out_f32 = self.device.new_storage(DType::F32, ids_rows * n)?;
        k::gemm(
            self.device.q(),
            k::SyclDType::F32,
            false,
            true,
            1,
            n as i64,
            k as i64,
            1.0,
            0.0,
            a.buf(),
            b.buf(),
            out_f32.buf(),
            ids_rows as i64,
            k as i64,
            // oneMKL batch strides are element counts, not bytes.
            (n * k) as i64,
            n as i64,
            0,
            0,
        )
        .map_err(werr)?;
        Ok((out_f32, Shape::from((batch, topk, n))))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &SyclDevice,
    data: &[T],
) -> Result<QStorage> {
    let bytes = std::mem::size_of_val(data);
    let host = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
    let buf = device.alloc_bytes(bytes)?;
    buf.copy_from_host(host)
        .map_err(|e| crate::Error::Sycl(SyclError::msg(e.to_string()).into()))?;
    Ok(QStorage::Sycl(QSyclStorage {
        data: buf,
        dtype: T::DTYPE,
        elem_count: data.len() * T::DTYPE.block_size(),
        device: device.clone(),
        dequant_cache: parking_lot::Mutex::new(None),
    }))
}
