//! Indexed MoE on CUDA: grouped quantized matmul for prefill, matvec for small
//! batches.
//!
//! The host side of `rocm/moe.rs`'s grouped path, over the same kernels in
//! `candle-kernels/src/quantized.cu`: `moe_group_pairs` packs the routed pairs
//! by expert, and `grouped_mul_mat_q5_K`/`_q6_K` run the dense MMQ tiles with
//! indirect input columns and output rows. The activations are requantized to
//! q8_1 exactly as the matvec path does, and the Q5 minimum correction keeps
//! the matvec's quantized sums, so the two paths differ only in accumulation
//! order. Only expert counts and an invalid-ID status cross to the host.
//!
//! nvcc builds `quantized.cu` with the AMPERE tile set (no RDNA define), which
//! is also what [`super::mul_mat_via_q8_1`] launches for the dense kernels.

use super::{pad, quantize_q8_1, QCudaStorage, MATRIX_ROW_PADDING, WARP_SIZE};
use crate::backend::BackendDevice;
use crate::cuda_backend::WrapErr;
use crate::quantized::{GgmlDType, QStorage, QTensor};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Layout, Result, Shape, Storage, Tensor};
use cudarc::driver::{CudaSlice, CudaView, PushKernelArg};

/// Immutable GPU-packed token/expert assignments, reusable across the
/// projections of one MoE layer. Captured at construction: later changes to
/// the source IDs do not reach it. Local to one routing decision, not a cache.
pub struct GroupedMoeRouting {
    ids: Tensor, // Kept for the shape and device checks; never re-read for routing.
    packed: PackedRouting,
}

impl std::fmt::Debug for GroupedMoeRouting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedMoeRouting")
            .field("num_experts", &self.packed.num_experts)
            .field("batch", &self.packed.batch)
            .field("topk", &self.packed.topk)
            .finish_non_exhaustive()
    }
}

impl GroupedMoeRouting {
    pub fn new(ids: &Tensor, num_experts: usize) -> Result<Self> {
        let packed = match &*ids.storage() {
            Storage::Cuda(storage) => PackedRouting::new(storage, ids.layout(), num_experts)?,
            _ => crate::bail!("grouped MoE routing requires CUDA IDs"),
        };
        Ok(Self {
            ids: ids.clone(),
            packed,
        })
    }

    /// `weights` (`(num_experts, n, k)`, Q5K or Q6K) applied to `x`
    /// (`(batch, 1 or topk, k)` f32) along this routing: `(batch, topk, n)`.
    pub fn forward(&self, weights: &QTensor, x: &Tensor) -> Result<Tensor> {
        match (&weights.storage, &*x.storage(), &*self.ids.storage()) {
            (QStorage::Cuda(q), Storage::Cuda(input), Storage::Cuda(ids)) => {
                let (storage, shape) = forward_impl(
                    q,
                    weights.shape(),
                    input,
                    x.layout(),
                    ids,
                    self.ids.layout(),
                    Some(true),
                    Some(&self.packed),
                )?;
                Ok(crate::tensor::from_storage(
                    Storage::Cuda(storage),
                    shape,
                    crate::op::BackpropOp::none(),
                    false,
                ))
            }
            _ => crate::bail!("prepared grouped MoE requires CUDA weights and input"),
        }
    }
}

struct PackedRouting {
    device: CudaDevice,
    num_experts: usize,
    batch: usize,
    topk: usize,
    max_count: usize,
    pairs: CudaSlice<u32>,
    counts: CudaSlice<u32>,
}

/// The shapes a launch is derived from, once validated.
pub(super) struct Dims {
    pub(super) num_experts: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) batch: usize,
    /// `1` when every expert of a token shares one activation row, `topk` when
    /// each routed pair carries its own.
    pub(super) input_dim1: usize,
    pub(super) topk: usize,
}

fn dims(self_shape: &Shape, input_l: &Layout, ids_l: &Layout) -> Result<Dims> {
    let (num_experts, n, k) = self_shape.dims3()?;
    let (batch, input_dim1, input_k) = input_l.shape().dims3()?;
    let (ids_batch, topk) = ids_l.shape().dims2()?;
    if input_k != k {
        crate::bail!("indexed_moe_forward: weights are {self_shape:?} but the input has k={input_k}")
    }
    if ids_batch != batch {
        crate::bail!("indexed_moe_forward: input batch {batch} but ids batch {ids_batch}")
    }
    if input_dim1 != 1 && input_dim1 != topk {
        crate::bail!("indexed_moe_forward: input dim 1 is {input_dim1}, expected 1 or topk {topk}")
    }
    if num_experts == 0 || topk == 0 || batch == 0 || n == 0 || k == 0 {
        crate::bail!("indexed_moe_forward: empty shape {self_shape:?} / {:?}", ids_l.shape())
    }
    Ok(Dims {
        num_experts,
        n,
        k,
        batch,
        input_dim1,
        topk,
    })
}

/// `(mmq_x, mmq_y, nwarps)` of `quantized.cu`'s AMPERE set, the one nvcc
/// compiles (`CUDA_USE_TENSOR_CORES` is never defined for it).
fn tiles(dtype: GgmlDType) -> Option<(usize, usize, usize)> {
    match dtype {
        GgmlDType::Q5K => Some((64, 128, 4)),
        GgmlDType::Q6K => Some((64, 64, 4)),
        _ => None,
    }
}

/// Keep decode and short or sparse suffixes on the vector path: eight routed
/// columns per expert, the ROCm rule. A dispatch rule, never a retry after a
/// grouped failure. K must be a whole number of the MMQ k-step (256 for Q5K
/// and Q6K, `qk * WARP_SIZE / qi`).
pub(super) fn use_grouped(dtype: GgmlDType, d: &Dims) -> bool {
    tiles(dtype).is_some()
        && d.batch > 1
        && d.batch * d.topk / d.num_experts >= 8
        && d.k.is_multiple_of(256)
}

pub(super) fn supports(q: &QCudaStorage, shape: &Shape, batch: usize, topk: usize) -> bool {
    let Ok((num_experts, n, k)) = shape.dims3() else {
        return false;
    };
    num_experts > 0
        && num_experts <= 65535
        && n > 0
        && topk > 0
        && batch.checked_mul(topk).is_some_and(|p| p <= i32::MAX as usize)
        && use_grouped(
            q.dtype,
            &Dims {
                num_experts,
                n,
                k,
                batch,
                input_dim1: 1,
                topk,
            },
        )
}

/// `q` is `(num_experts, n, k)`, `input` is `(batch, topk or 1, k)` f32 and
/// `ids` is `(batch, topk)` u32. Returns `(batch, topk, n)` f32.
pub(super) fn forward(
    q: &QCudaStorage,
    self_shape: &Shape,
    input: &CudaStorage,
    input_l: &Layout,
    ids: &CudaStorage,
    ids_l: &Layout,
) -> Result<(CudaStorage, Shape)> {
    forward_impl(q, self_shape, input, input_l, ids, ids_l, None, None)
}

fn contiguous<'a, T: cudarc::driver::DeviceRepr>(
    slice: &'a CudaSlice<T>,
    l: &Layout,
    len: usize,
    what: &str,
) -> Result<CudaView<'a, T>> {
    match l.contiguous_offsets() {
        Some((start, end)) if end - start == len => Ok(slice.slice(start..end)),
        _ => crate::bail!("indexed_moe_forward expects contiguous {what}, got {l:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_impl(
    q: &QCudaStorage,
    self_shape: &Shape,
    input: &CudaStorage,
    input_l: &Layout,
    ids: &CudaStorage,
    ids_l: &Layout,
    grouped_override: Option<bool>,
    prepared: Option<&PackedRouting>,
) -> Result<(CudaStorage, Shape)> {
    if !q.device.same_device(&input.device) || !q.device.same_device(&ids.device) {
        crate::bail!("indexed_moe_forward: weights, input and ids must share one CUDA device/stream")
    }
    let d = dims(self_shape, input_l, ids_l)?;
    let y = contiguous(
        input.as_cuda_slice::<f32>()?,
        input_l,
        d.batch * d.input_dim1 * d.k,
        "f32 input",
    )?;
    let ids_view = contiguous(ids.as_cuda_slice::<u32>()?, ids_l, d.batch * d.topk, "u32 ids")?;
    let grouped = grouped_override.unwrap_or_else(|| use_grouped(q.dtype, &d));
    if !grouped {
        return super::indexed_moe_forward_fused_q8_1_input(
            &q.data.inner.slice(0..),
            self_shape,
            q.dtype,
            &y,
            input_l.shape(),
            &ids_view,
            ids_l.shape(),
            &q.device,
        );
    }
    let owned;
    let routing = match prepared {
        Some(p) => p,
        None => {
            owned = PackedRouting::new(ids, ids_l, d.num_experts)?;
            &owned
        }
    };
    let out = grouped_forward(q, &d, &y, routing)?;
    Ok((
        CudaStorage::wrap_cuda_slice(out, q.device.clone()),
        (d.batch, d.topk, d.n).into(),
    ))
}

impl PackedRouting {
    fn new(ids: &CudaStorage, ids_l: &Layout, num_experts: usize) -> Result<Self> {
        let (batch, topk) = ids_l.shape().dims2()?;
        let total_pairs = batch
            .checked_mul(topk)
            .filter(|&n| n > 0 && n <= i32::MAX as usize)
            .ok_or_else(|| crate::Error::Msg("invalid grouped MoE route dimensions".into()))?;
        if num_experts == 0 || num_experts > 65535 {
            crate::bail!("invalid grouped MoE expert count")
        }
        let ids_view = match (ids.as_cuda_slice::<u32>(), ids_l.contiguous_offsets()) {
            (Ok(s), Some((start, end))) if end - start == total_pairs => s.slice(start..end),
            _ => crate::bail!("grouped MoE routing requires contiguous u32 IDs"),
        };
        let dev = &ids.device;
        let scratch_len = num_experts
            .checked_mul(total_pairs)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| crate::Error::Msg("grouped MoE routing scratch overflow".into()))?;
        let pairs = dev.alloc_zeros::<u32>(scratch_len)?;
        let counts = dev.alloc_zeros::<u32>(num_experts + 1)?;
        let func = dev.get_or_load_func("moe_group_pairs", &candle_kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_experts as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&ids_view);
        builder.arg(&pairs);
        builder.arg(&counts);
        barg!(builder, total_pairs as i32, num_experts as i32);
        unsafe { builder.launch(cfg) }.w()?;
        // A small metadata readback (33 u32 for LFM2.5), not the routing
        // tensor; it synchronizes the stream. Invalid IDs become an error
        // before any weight read, and the column grid is bounded by the
        // busiest expert, duplicate routes within a token included.
        let host_counts = dev.clone_dtoh(&counts)?;
        if host_counts[num_experts] != 0 {
            crate::bail!("grouped MoE: expert id is outside the weight stack")
        }
        let accounted: u64 = host_counts[..num_experts].iter().map(|n| *n as u64).sum();
        if accounted != total_pairs as u64 {
            crate::bail!("grouped MoE routing counts do not account for every pair")
        }
        let max_count = *host_counts[..num_experts].iter().max().unwrap() as usize;
        Ok(Self {
            device: dev.clone(),
            num_experts,
            batch,
            topk,
            max_count,
            pairs,
            counts,
        })
    }
}

fn grouped_forward(
    q: &QCudaStorage,
    d: &Dims,
    y: &CudaView<f32>,
    routing: &PackedRouting,
) -> Result<CudaSlice<f32>> {
    let name = match q.dtype {
        GgmlDType::Q5K => "grouped_mul_mat_q5_K",
        GgmlDType::Q6K => "grouped_mul_mat_q6_K",
        _ => crate::bail!("grouped MoE requires Q5K or Q6K weights, not {:?}", q.dtype),
    };
    let (mmq_x, mmq_y, nwarps) = tiles(q.dtype).expect("Q5K and Q6K have tiles");
    if !d.k.is_multiple_of(256) || d.num_experts > 65535 {
        crate::bail!("unsupported grouped MoE dimensions")
    }
    if routing.num_experts != d.num_experts
        || routing.batch != d.batch
        || routing.topk != d.topk
        || !routing.device.same_device(&q.device)
    {
        crate::bail!("grouped MoE routing does not match weight experts, input shape or device/stream")
    }
    // The kernel strides between experts by `n * k / block_size` blocks with
    // no bound of its own.
    let data_elems = q.data.len / q.dtype.type_size() * q.dtype.block_size();
    if data_elems < d.num_experts * d.n * d.k {
        crate::bail!("grouped MoE: weights hold {data_elems} elems, need {}", d.num_experts * d.n * d.k)
    }
    let column_tiles = routing.max_count.div_ceil(mmq_x);
    if column_tiles > 65535 {
        crate::bail!("grouped MoE column grid exceeds the CUDA grid-y limit")
    }
    let dev = &q.device;
    let total_rows = d.batch * d.input_dim1;
    let k_padded = pad(d.k, MATRIX_ROW_PADDING);
    let mut y_q8_1 = dev.alloc_zeros::<u8>(
        k_padded * total_rows * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size(),
    )?;
    quantize_q8_1(y, &mut y_q8_1, d.k, total_rows, dev)?;
    // Every routed pair's rows are written (the counts account for all of
    // them); zeroed anyway so a coverage bug reads as zeros, not garbage.
    let out = dev.alloc_zeros::<f32>(d.batch * d.topk * d.n)?;
    let func = dev.get_or_load_func(name, &candle_kernels::QUANTIZED)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            d.n.div_ceil(mmq_y) as u32,
            column_tiles.max(1) as u32,
            d.num_experts as u32,
        ),
        block_dim: (WARP_SIZE as u32, nwarps as u32, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&q.data.inner);
    builder.arg(&y_q8_1);
    builder.arg(&out);
    builder.arg(&routing.pairs);
    builder.arg(&routing.counts);
    barg!(
        builder,
        d.n as i32,
        d.k as i32,
        k_padded as i32,
        (d.batch * d.topk) as i32,
        d.topk as i32,
        d.input_dim1 as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(out)
}

#[cfg(test)]
fn forward_for_test(w: &QTensor, x: &Tensor, ids: &Tensor, grouped: bool) -> Result<Tensor> {
    match (&w.storage, &*x.storage(), &*ids.storage()) {
        (QStorage::Cuda(q), Storage::Cuda(x_s), Storage::Cuda(ids_s)) => {
            let (out, shape) =
                forward_impl(q, w.shape(), x_s, x.layout(), ids_s, ids.layout(), Some(grouped), None)?;
            Ok(crate::tensor::from_storage(
                Storage::Cuda(out),
                shape,
                crate::op::BackpropOp::none(),
                false,
            ))
        }
        _ => crate::bail!("test expects CUDA weights and tensors"),
    }
}

#[cfg(test)]
mod tests;
