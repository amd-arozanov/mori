// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Experiment-only helpers for the ep_ll-port ablation (branch exp/epll-port-ablation).
// Ports selected ep_ll_dynamic_bench transport primitives into the MORI IntraNodeLL
// dispatch/combine path so each optimization can be ablated independently via the
// MORI_LL_OPT env selector. NOT part of the production launch/tuning path.
//
//   P3: 128-bit paired (count, sequence) store + s_sleep spin backoff.
//   P1: per-record readiness flag poll (separate flag region; see design doc).
#pragma once

#include <stdint.h>

namespace mori {
namespace moe {
namespace epll {

// 128-bit vector for paired 16B stores/loads (ep_ll store_pair / load_pair).
typedef uint64_t U64x2 __attribute__((ext_vector_type(2)));

// Atomic 16B store: lo+hi become visible together (no torn read of count vs seq).
__device__ __forceinline__ void store_pair(volatile uint64_t* dst, uint64_t lo, uint64_t hi) {
  const U64x2 pair = {lo, hi};
  *reinterpret_cast<volatile U64x2*>(dst) = pair;
}

__device__ __forceinline__ U64x2 load_pair(const volatile uint64_t* src) {
  return *reinterpret_cast<const volatile U64x2*>(src);
}

// ep_ll uses sequence value itself as the ready flag.
__device__ __forceinline__ uint64_t ll_flag(uint64_t seq) { return seq; }

// Spin until *ptr >= want, with s_sleep(1) backoff (lower power/contention while
// polling a peer-written cache line). Returns 1 on success, 0 on timeout.
__device__ __forceinline__ int wait_u32_at_least(const volatile uint32_t* ptr, uint32_t want,
                                                 int poll_iters) {
  for (int spin = 0; spin < poll_iters; ++spin) {
    if (*ptr >= want) return 1;
    __builtin_amdgcn_s_sleep(1);
  }
  return 0;
}

// Default spin budget for the experiment (large; correctness-mode workloads complete
// well within this, matching ep_ll's poll_iters scale).
constexpr int kDefaultPollIters = 1000000;

// ---------------------------------------------------------------------------
// LL128 self-flagged line transport (full ep_ll alignment, MORI_LL_OPT=epll_full).
//
// Each 64B line = 8x u64 = 7 data words + 1 trailing flag word (u64[7]). The line is
// written by 4 consecutive lanes of ONE wave (pair = lane%4), each issuing a 16B
// store_pair; the wave coalesces the 4 sub-writes into a single cacheline transfer so
// "flag visible => whole line visible" WITHOUT a system fence (the ep_ll premise). The
// receiver self-gates per line on u64[7] == flag, then copies the 7 data words. This
// replaces MORI's direct-write + __threadfence_system + separate release flag.
// ---------------------------------------------------------------------------
constexpr int kLLPairsPerLine = 4;       // 4 lanes cooperate per line
constexpr int kLLDataWordsPerLine = 7;   // 7 data u64 + 1 flag u64 = 64B
constexpr int kLLLineU64 = 8;
constexpr int kLLLineBytes = 64;
constexpr uint64_t kLLReadyFlag = 1;     // nonzero ready sentinel (reset to 0 for reuse)

__host__ __device__ __forceinline__ size_t LLLinesPerToken(size_t hiddenBytes) {
  const size_t words = (hiddenBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  return (words + kLLDataWordsPerLine - 1) / kLLDataWordsPerLine;
}

__host__ __device__ __forceinline__ size_t LLStrideBytes(size_t hiddenBytes) {
  return LLLinesPerToken(hiddenBytes) * (size_t)kLLLineBytes;
}

// Sender: pack `hiddenWords` u64 of `hidden` into self-flagged LL128 lines at `lineBase`
// (remote staging slot). Cooperative over [0,groupThreads); thread t -> (line=t/4, pair=t%4).
// groupThreads must be a multiple of 4 with each 4-lane group inside one wave.
__device__ __forceinline__ void LLWriteToken(uint8_t* lineBase, const uint64_t* hidden,
                                             size_t hiddenWords, uint64_t flag,
                                             int threadInGroup, int groupThreads) {
  const size_t numLines = (hiddenWords + kLLDataWordsPerLine - 1) / kLLDataWordsPerLine;
  const int group = threadInGroup / kLLPairsPerLine;
  const int pair = threadInGroup % kLLPairsPerLine;
  const int lineStep = groupThreads / kLLPairsPerLine;
  uint64_t* lines = reinterpret_cast<uint64_t*>(lineBase);
  for (size_t line = group; line < numLines; line += lineStep) {
    const size_t w0 = line * kLLDataWordsPerLine + (size_t)pair * 2;
    const uint64_t lo = (w0 < hiddenWords) ? hidden[w0] : 0ull;
    uint64_t hi;
    if (pair == kLLPairsPerLine - 1) {
      hi = flag;  // last pair carries data word 6 (lo) + flag (hi)
    } else {
      hi = (w0 + 1 < hiddenWords) ? hidden[w0 + 1] : 0ull;
    }
    store_pair(&lines[line * kLLLineU64 + (size_t)pair * 2], lo, hi);
  }
}

// Receiver: copy `hiddenWords` u64 from self-flagged LL128 lines at `lineBase` (local
// staging) into the clean contiguous `dst`. Cooperative over a warp [0,warpThreads);
// each lane self-gates on its line's flag (no __syncwarp: every lane spins on the same
// u64[7], then reads its own pair from the now-present cacheline). Resets the flag to 0.
__device__ __forceinline__ void LLReadToken(const uint8_t* lineBase, uint64_t* dst,
                                            size_t hiddenWords, uint64_t flag, int lane,
                                            int warpThreads, int pollIters) {
  const size_t numLines = (hiddenWords + kLLDataWordsPerLine - 1) / kLLDataWordsPerLine;
  const int group = lane / kLLPairsPerLine;
  const int pair = lane % kLLPairsPerLine;
  const int lineStep = warpThreads / kLLPairsPerLine;
  uint64_t* lines = reinterpret_cast<uint64_t*>(const_cast<uint8_t*>(lineBase));
  for (size_t line = group; line < numLines; line += lineStep) {
    volatile uint64_t* L = lines + line * kLLLineU64;
    for (int spin = 0; spin < pollIters; ++spin) {
      if (L[7] == flag) break;
      __builtin_amdgcn_s_sleep(1);
    }
    const size_t w0 = line * kLLDataWordsPerLine + (size_t)pair * 2;
    if (w0 < hiddenWords) dst[w0] = L[(size_t)pair * 2];
    if (pair != kLLPairsPerLine - 1 && (w0 + 1) < hiddenWords) {
      dst[w0 + 1] = L[(size_t)pair * 2 + 1];
    }
    if (pair == kLLPairsPerLine - 1) L[7] = 0;  // reset for next-iter reuse
  }
}

}  // namespace epll
}  // namespace moe
}  // namespace mori
