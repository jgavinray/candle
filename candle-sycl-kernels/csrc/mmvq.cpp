// Fused quantized mat-vec (QMatMul for small batch, the decode path). The
// weight stays quantized in memory.
//
// Two paths:
//  - `candle_sycl_mmvq_q8` (preferred): the activation is quantized to int8
//    once per call (per-256 for K-quants, as the CPU `BlockQ8K::from_float`;
//    per-32 for Q4_0/Q8_0) and each weight block is dotted against it in
//    integer arithmetic, matching the CPU `vec_dot` numerics.
//  - `candle_sycl_mmvq`: dequantize each block to f32 and float-dot; the
//    fallback for types without an integer kernel.
//
// Both launch one work-item per (output row, chunk of `ch` weight blocks),
// which writes a partial sum to `tmp`, then one per output row to sum its
// chunks. `ch` is picked by the caller to bound `tmp`; 1 is fastest.
#include "quant_blocks.hpp"

namespace {
constexpr int MAX_M = 8; // batch bound of the float path

// `item(mi, row, b)` -> the dot of weight block `row*nblk + b` against
// activation row `mi`. Writes `out[mi*n + row]`.
template <typename Item>
void launch_chunked(sycl::queue &q, size_t n, size_t m, size_t nblk, size_t ch, float *tmp,
                    void *out, bool out_f16, Item item) {
  size_t nch = (nblk + ch - 1) / ch;
  // A 3D range rather than folding (mi, row) into one index: Xe has no hardware
  // integer divide, so recovering them with `/ n` and `% n` costs more than the
  // dot product it indexes for. `ch == 1` gets its own body so the dot is not
  // buried in a loop whose trip count the compiler cannot see.
  if (ch == 1) {
    q.parallel_for(sycl::range<3>(m, n, nch), [=](sycl::id<3> id) {
      size_t mi = id[0], row = id[1], c = id[2];
      tmp[(mi * n + row) * nch + c] = item(mi, row, c);
    });
  } else {
    q.parallel_for(sycl::range<3>(m, n, nch), [=](sycl::id<3> id) {
      size_t mi = id[0], row = id[1], c = id[2];
      size_t b1 = sycl::min((c + 1) * ch, nblk);
      float acc = 0.f;
      for (size_t b = c * ch; b < b1; ++b) acc += item(mi, row, b);
      tmp[(mi * n + row) * nch + c] = acc;
    });
  }
  q.parallel_for(sycl::range<2>(m, n), [=](sycl::id<2> id) {
    size_t mr = id[0] * n + id[1];
    float acc = 0.f;
    for (size_t c = 0; c < nch; ++c) acc += tmp[mr * nch + c];
    if (out_f16)
      static_cast<f16 *>(out)[mr] = (f16)acc;
    else
      static_cast<float *>(out)[mr] = acc;
  });
}

template <typename Blk, int BLK, typename Deq>
int run_mmvq(CandleSyclQueue *q, const void *w, const float *act, float *out, size_t n,
             size_t k, size_t m, float *tmp, size_t ch, Deq deq) {
  const Blk *wb = static_cast<const Blk *>(w);
  size_t nblk = k / BLK;
  try {
    launch_chunked(q->q, n, m, nblk, ch, tmp, out, false, [=](size_t mi, size_t row, size_t b) {
      float blk[BLK];
      deq(wb[row * nblk + b], blk);
      const float *a = act + mi * k + b * BLK;
      float s = 0.f;
      for (int j = 0; j < BLK; ++j) s += blk[j] * a[j];
      return s;
    });
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_LAUNCH;
  }
}

#include "mmvq_internal.hpp"  // quantize_act + dot_blk, shared with indexed_moe.cpp
template <typename Blk, int BLK, typename Dot>
int run_mmvq_q8(CandleSyclQueue *cq, const void *w, const void *act, bool act_f16, void *out,
                bool out_f16, size_t n, size_t k, size_t m, int8_t *q8, float *d8, int32_t *s32,
                float *tmp, size_t ch, Dot dot) {
  const Blk *wb = static_cast<const Blk *>(w);
  size_t nblk = k / BLK;
  try {
    quantize_act(cq->q, BLK, act, act_f16, k, m, q8, d8, s32);
    launch_chunked(cq->q, n, m, nblk, ch, tmp, out, out_f16, [=](size_t mi, size_t row, size_t b) {
      size_t yb = mi * nblk + b;
      return dot(wb[row * nblk + b], q8 + yb * BLK, d8[yb], s32 + yb * 8);
    });
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_LAUNCH;
  }
}
} // namespace

// Integer mat-vec. `act` / `out` are f32, or f16 when `act_f16` / `out_f16`.
// Scratch: `q8` holds `m*k` bytes, `d8` `m*(k/blk)` f32s, `s32` `m*(k/blk)*8`
// i32s (blk = 256 for K-quants, 32 otherwise), `tmp` `m*n*ceil((k/blk)/ch)`
// f32s. Returns CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE for types without an integer
// kernel; the caller then uses `candle_sycl_mmvq`.
extern "C" int candle_sycl_mmvq_q8(CandleSyclQueue *q, uint32_t dt, const void *w,
                                   const void *act, int act_f16, void *out, int out_f16,
                                   size_t n, size_t k, size_t m, int8_t *q8, float *d8,
                                   int32_t *s32, float *tmp, size_t ch) {
  const bool af = act_f16 != 0, of = out_f16 != 0;
  // `dot_blk` is overloaded on the block type, so one lambda serves every case.
  auto dot = [](const auto &x, const int8_t *y, float d, const int32_t *s) {
    return dot_blk(x, y, d, s);
  };
  switch (dt) {
  case G_Q4_0: return run_mmvq_q8<BQ4_0, QK>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  case G_Q8_0: return run_mmvq_q8<BQ8_0, QK>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  case G_Q4K: return run_mmvq_q8<BQ4K, QK_K>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  case G_Q5K: return run_mmvq_q8<BQ5K, QK_K>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  case G_Q6K: return run_mmvq_q8<BQ6K, QK_K>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}

// Float mat-vec; `tmp` as for `candle_sycl_mmvq_q8`.
extern "C" int candle_sycl_mmvq(CandleSyclQueue *q, uint32_t dt, const void *w, const float *act,
                                float *out, size_t n, size_t k, size_t m, float *tmp,
                                size_t ch) {
  if (m > MAX_M) return CANDLE_SYCL_ERR_INVALID;
  return dispatch_blk(dt, [&](auto tag) {
    return run_mmvq<typename decltype(tag)::Blk, decltype(tag)::blk>(
        q, w, act, out, n, k, m, tmp, ch,
        [](const auto &b, float *y) { deq_blk(b, y); });
  });
}
