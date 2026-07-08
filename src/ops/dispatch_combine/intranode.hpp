// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <type_traits>

#include "mori/core/core.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/core/profiler/kernel_profiler.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/common.hpp"
#include "src/ops/dispatch_combine/convert.hpp"
#include "src/ops/dispatch_combine/ep_ll_helpers.hpp"
#ifdef ENABLE_PROFILER
#include "mori/profiler/profiler.hpp"
#endif

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierIntraNodeKernel(EpDispatchCombineArgs<T> args,
                                                         const uint64_t crossDeviceBarrierFlag) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  __syncthreads();
  if (thdId == 0) atomicAdd(args.combineGridBarrier, 1);

  if (globalThdId < args.config.worldSize) {
    // Set remote flag after all copies are done
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, gridDim.x);
    __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    __threadfence_system();
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(globalThdId) + args.config.rank,
        crossDeviceBarrierFlag);
  }

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchIntraNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool EnableStdMoE = false, bool EnableSnooze = false>
__device__ void EpDispatchIntraNodeKernel_body(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;
  size_t hiddenDim = config.HiddenDimSz();

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::DispatchSendTokens);

  if (args.tokenIndices && args.inpTokenBuf) {
    // Phase1: send token
    // Each warp compute token offset on destinition PE
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
         i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destExpert = args.tokenIndices[i];
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t destTokId = 0;

      // Out-of-range expert id guard: destPe is warp-uniform here (one
      // token-expert per warp) and indexes GetAs(destPe) / destPeTokenCounter
      // below. An out-of-range id (e.g. an EPLB physical id
      // >= worldSize*numExpertPerRank) would index those out of bounds (the
      // assert at dispatch is stripped under NDEBUG) -> HSA page fault. Drop it
      // via the same overflow sentinel the dedup path uses; the whole warp
      // skips coherently.
      if ((destPe < 0) || (destPe >= config.worldSize)) {
        if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
        continue;
      }

      // Deduplicate
      assert(config.numExpertPerToken < warpSize);
      int condition = 0;
      if (laneId < (i % config.numExpertPerToken)) {
        condition = destPe == (args.tokenIndices[srcTokId * config.numExpertPerToken + laneId] /
                               config.numExpertPerRank);
      }
      if (__any(condition)) {
        // Indicate that this token is already sent to the destination PE by setting an overflow
        // token index
        if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
        continue;
      }

      if (laneId == 0) {
        // decide token id in dest pe
        destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
        assert(destTokId < config.MaxNumTokensToRecv() &&
               "Total recv token overflow: increase maxTotalRecvTokens");
        atomicAdd(args.destPeTokenCounter + destPe, 1);
        // In dispDestTokIdMap, record the destination slot for this token-expert pair (flat index
        // into the dest PE's recv buffer) In dispTokIdToSrcTokIdMemObj on the dest PE, record which
        // global source token occupies this slot (for combine-phase routing)
        args.dispDestTokIdMap[i] = FlatTokenIndex(config, destPe, destTokId);
        args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
            FlatTokenIndex(config, myPe, srcTokId);
      }
      destTokId = __shfl(destTokId, 0);

      // Write weights and indices
      if (laneId < config.numExpertPerToken) {
        if (args.weightsBuf) {
          args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
              destPe)[destTokId * config.numExpertPerToken + laneId] =
              args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
        }
        args.shmemOutIndicesMemObj->template GetAs<index_t*>(
            destPe)[destTokId * config.numExpertPerToken + laneId] =
            args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
      }

      // Write scales
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        size_t destScaleOffset = (size_t)destTokId * config.scaleDim * config.scaleTypeSize;
        size_t srcScaleOffset = (size_t)srcTokId * config.scaleDim * config.scaleTypeSize;
        core::WarpCopy(
            args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
            args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
      }

      size_t srcTokOffset = srcTokId * hiddenDim;
      size_t destTokOffset = destTokId * hiddenDim;

      core::WarpCopy(args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
                     args.inpTokenBuf + srcTokOffset, hiddenDim);
    }
  }
  __syncthreads();
  if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Send token num & token to expert mapping to other ranks
  MORI_TRACE_NEXT(seq, Slot::DispatchNotifyPeer);
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      if constexpr (EnableSnooze) {
        // [exp/epll-port] P3: s_sleep backoff on the cross-device reuse spin (ep_ll snooze).
        while (core::AtomicLoadRelaxedSystem(signal) != 0) {
          __builtin_amdgcn_s_sleep(1);
        }
      } else {
        shmem::ShmemInt32WaitUntilEquals(signal, 0);
      }
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  MORI_TRACE_NEXT(seq, Slot::DispatchWaitPeerToken);
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum;
      if constexpr (EnableSnooze) {
        // [exp/epll-port] P3: s_sleep backoff on the cross-device count-wait spin (ep_ll snooze).
        index_t v;
        while ((v = core::AtomicLoadRelaxedSystem(signal)) <= 0) {
          __builtin_amdgcn_s_sleep(1);
        }
        recvTokenNum = v - 1;
      } else {
        recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      }
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);

      // reset local counter
      args.destPeTokenCounter[destPe] = 0;
    }

    // reset counter
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
  }

#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

template <typename T, bool EnableStdMoE = false, bool EnableSnooze = false,
          bool EnableSPSC = false, bool EnableLL128 = false, bool EnableEpll = false,
          bool EnableEpllNoMeta = false, bool EnableEpllLine = false>
__device__ void EpDispatchIntraNodeLLKernel_body(EpDispatchCombineArgs<T> args) {
  // [exp/epll-port] epll_full (LL128 self-flagged lines) carries only hidden in staging and routes
  // metadata via the deferred remote scalar writes (base-style), exactly like the nometa path.
  constexpr bool kEpllDeferMeta = EnableEpllNoMeta || EnableEpllLine;
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;
  size_t hiddenDim = config.HiddenDimSz();
  const bool hasScales = args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0);

  // Warp-group coordination: 2 warps per group
  constexpr int kWarpsPerGroup = 2;
  constexpr int kMaxWarpGroups = 8;

  __shared__ uint64_t groupData[kMaxWarpGroups];  // (iteration+1, destTokId)
  __shared__ int groupCounters[kMaxWarpGroups];   // consume counter
  // [exp/epll-port] epll send: monotonic per-group "copy done" epoch. Every warp in the group
  // bumps it after copying+fencing its hidden chunk; the header warp waits on it before
  // publishing the per-slot release flag, so flag-visible => all warps' data visible.
  __shared__ int groupCopyEpoch[kMaxWarpGroups];

  int warpGroupIdInBlock = warpId / kWarpsPerGroup;
  int inGroupWarpId = warpId % kWarpsPerGroup;
  int warpGroupId = globalWarpId / kWarpsPerGroup;
  int warpGroupNum = globalWarpNum / kWarpsPerGroup;

  // clear shared memory
  if (inGroupWarpId == 0 && laneId == 0) {
    groupData[warpGroupIdInBlock] = 0;
    groupCounters[warpGroupIdInBlock] = kWarpsPerGroup - 1;
    groupCopyEpoch[warpGroupIdInBlock] = 0;
  }
  __syncthreads();

  // Hidden dim split for each warp in group
  size_t dimPerWarp = (hiddenDim + kWarpsPerGroup - 1) / kWarpsPerGroup;
  size_t warpDimOffset = (size_t)inGroupWarpId * dimPerWarp;
  size_t warpDimChunk =
      (warpDimOffset < hiddenDim) ? min(hiddenDim - warpDimOffset, dimPerWarp) : 0;
  assert((warpNum % kWarpsPerGroup == 0) && (warpDimChunk > 0) &&
         "total num of warps must be divisible by the num of warpgroups, "
         "warpDimChunk must be > 0 for warpgroups to be useful");

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);

  if constexpr (EnableLL128 || EnableEpll) {
    // [exp/epll-port] LL128 upfront count: the per-dst send count is fully determined by the
    // routing table (tokenIndices) + the same dedup as the send loop, so warp0 computes it
    // locally and signals each peer EARLY (no grid barrier). The receiver then learns its
    // per-source counts cheaply and gates token arrival via per-token flags below.
    MORI_TRACE_NEXT(seq, Slot::DispatchNotifyPeer);
    if (globalWarpId == 0 && args.tokenIndices) {
      // Each lane owns one destination pe and counts how many of this rank's tokens route to it
      // (once per token == the send-loop dedup), then signals that peer early.
      index_t destPe = laneId;
      if (destPe < npes) {
        index_t myCount = 0;
        for (int t = 0; t < args.curRankNumToken; ++t) {
          for (int e = 0; e < config.numExpertPerToken; ++e) {
            index_t dst =
                args.tokenIndices[t * config.numExpertPerToken + e] / config.numExpertPerRank;
            if (dst == destPe) {
              ++myCount;  // first (and only counted) occurrence for this token
              break;
            }
          }
        }
        index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
        // reuse-spin: wait until the previous iteration's count was consumed (reset to 0).
        shmem::ShmemInt32WaitUntilEquals(signal, 0);
        // +1 so a zero count is still distinguishable from "not yet sent".
        core::AtomicStoreReleaseSystem(signal, myCount + 1);
      }
    }
  }

  MORI_TRACE_NEXT(seq, Slot::DispatchSendTokens);

  // [exp/epll-port] epll: local count of this group's non-skipped iterations, scaled by
  // kWarpsPerGroup. The header warp waits until groupCopyEpoch reaches this before flagging.
  [[maybe_unused]] int epllSendEpoch = 0;

  if (args.tokenIndices && args.inpTokenBuf) {
    // Phase1: send token
    // Each warp-group (4 warps) processes one token-expert pair
    for (int i = warpGroupId; i < args.curRankNumToken * config.numExpertPerToken;
         i += warpGroupNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destExpert = args.tokenIndices[i];
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t destTokId = 0;

      // prefetch remote addr (unused under SPSC: deterministic local slot, no remote RMW)
      [[maybe_unused]] auto dispTokOffset =
          args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe);

      // ALL warps in warp-group do dedup independently (same input = same result)
      assert(config.numExpertPerToken < warpSize);
      int condition = 0;
      if (laneId < (i % config.numExpertPerToken)) {
        condition = destPe == (args.tokenIndices[srcTokId * config.numExpertPerToken + laneId] /
                               config.numExpertPerRank);
      }
      if (__any(condition)) {
        // All 4 warps skip together, only warp 0 writes the skip marker
        if (inGroupWarpId == 0 && laneId == 0) {
          args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
        }
        continue;
      }

      // prefetch remote addr (unused under LL128: token goes to combineInp staging, then local deflag)
      [[maybe_unused]] auto dispatchOut = args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe);

      // Header Warp: atomic allocation + publish destTokId via shared memory.
      // EXP1 (reserve/copy decouple): publish destTokId ASAP, then issue the 16KB
      // WarpCopy first; the dependent remote metadata scalar writes are deferred to
      // AFTER the copy so their per-transaction issue latency overlaps the copy.
      if (inGroupWarpId == 0) {
        if (laneId == 0) {
          // Atomic allocation
          MORI_TRACE_NEXT(seq, Slot::DispatchReserveAtomic);
          if constexpr (EnableSPSC || EnableEpll) {
            // [exp/epll-port] SPSC: deterministic source-owned slot, no remote atomic.
            // Reuse destPeTokenCounter as the local per-dest index (0,1,2,... == final
            // count, so the notify-phase count signal stays correct). Recv buffer is
            // partitioned by source: slot in [srcPe*MaxRecvPerRank, (srcPe+1)*MaxRecvPerRank).
            index_t localIdx = atomicAdd(args.destPeTokenCounter + destPe, 1);
            destTokId = myPe * config.MaxNumTokensToRecvPerRank() + localIdx;
            assert(localIdx < config.MaxNumTokensToRecvPerRank() && "SPSC lane overflow");
          } else {
            destTokId = atomicAdd(dispTokOffset, 1);
            assert(destTokId < config.MaxNumTokensToRecv() &&
                   "Total recv token overflow: increase maxTotalRecvTokens");
          }

          // Wait for all consumers done (counter == N-1), then set to 0
          MORI_TRACE_NEXT(seq, Slot::DispatchReserveHandshake);
          while (atomicCAS(&groupCounters[warpGroupIdInBlock], kWarpsPerGroup - 1, 0) !=
                 kWarpsPerGroup - 1) {
          }
          // Write to shared mem: use (i+1) to avoid confusion with initial value 0
          MORI_TRACE_NEXT(seq, Slot::DispatchReserveWrites);
          __hip_atomic_store((unsigned long long*)&groupData[warpGroupIdInBlock],
                             ((uint64_t)(i + 1) << 32) | (uint32_t)destTokId, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_WORKGROUP);
        }
        destTokId = __shfl(destTokId, 0);
      } else {
        // Normal Warps: spin wait for iteration match, then consume
        // Only lane 0 spins, then broadcast to other lanes
        if (laneId == 0) {
          uint64_t val;
          do {
            val = __hip_atomic_load(
                reinterpret_cast<unsigned long long*>(&groupData[warpGroupIdInBlock]),
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
          } while ((val >> 32) != (uint32_t)(i + 1));
          atomicAdd(&groupCounters[warpGroupIdInBlock], 1);
          destTokId = (index_t)(val & 0xFFFFFFFF);
        }

        destTokId = __shfl(destTokId, 0);
      }

      // All warps in warpgroup: copy their portion of token data
      size_t srcTokOffset = srcTokId * hiddenDim;
      [[maybe_unused]] size_t destTokOffset = destTokId * hiddenDim;

      MORI_TRACE_NEXT(seq, Slot::DispatchCopyToken);
      if constexpr (EnableEpll && EnableEpllLine) {
        // [exp/epll-port] epll_full: write the token as self-flagged LL128 lines into the peer's
        // staging slot (7 data words + 1 seq flag per 64B line, one wave per 4-lane group). No
        // __threadfence_system, no group-epoch, no separate release flag: each line self-gates the
        // receiver. Metadata goes via the deferred remote scalar writes below (kEpllDeferMeta).
        const size_t hiddenBytes = config.HiddenBytes(sizeof(T));
        const size_t hiddenWords = hiddenBytes / sizeof(uint64_t);
        const size_t llStride = epll::LLStrideBytes(hiddenBytes);
        uint8_t* lineBase = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                            (size_t)destTokId * llStride;
        const uint64_t* srcWords =
            reinterpret_cast<const uint64_t*>(args.inpTokenBuf + srcTokOffset);
        epll::LLWriteToken(lineBase, srcWords, hiddenWords, epll::kLLReadyFlag,
                           inGroupWarpId * warpSize + laneId, kWarpsPerGroup * warpSize);
      } else if constexpr (EnableEpll) {
        // [exp/epll-port] epll: the WHOLE warp-group writes the token DIRECTLY into the final
        // dispatchOut slot on the peer (warp-group split, no staging, no deflag pass), matching
        // the base path's per-warp bandwidth. Each warp fences its own chunk to system scope and
        // bumps the group's copy epoch; the header warp waits for the group, then publishes a
        // release flag in the combineInp staging slot's flag word so flag-visible => data-visible.
        // The receiver only WAITS on the flag (multi-block, below); it does not copy.
        core::WarpCopy<T, 2>(dispatchOut + destTokOffset + warpDimOffset,
                             args.inpTokenBuf + srcTokOffset + warpDimOffset, warpDimChunk);
        __threadfence_system();
        if (laneId == 0) {
          __hip_atomic_fetch_add(&groupCopyEpoch[warpGroupIdInBlock], 1, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_WORKGROUP);
        }
        if (inGroupWarpId == 0) {
          epllSendEpoch += kWarpsPerGroup;
          if (laneId == 0) {
            while (__hip_atomic_load(&groupCopyEpoch[warpGroupIdInBlock], __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_WORKGROUP) < epllSendEpoch) {
            }
            uint8_t* stageBase =
                args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                (size_t)destTokId * config.MaxXferBytesPerToken();
            // [exp/epll-port] Step B: metadata-in-line. Pack {srcTokFlat, weights, indices} into a
            // single contiguous region at the staging slot start (hidden goes straight to
            // dispatchOut, so [0,HiddenBytes) is free) instead of 1+2*topk separate remote scalar
            // writes. The release-flag store below orders these writes; the receiver unpacks locally.
            // [exp/epll-port] EnableEpllNoMeta ablation: skip the inline pack entirely and fall back
            // to the deferred remote scalar metadata writes below (base-style), keeping only the
            // direct dispatchOut write + per-slot release flag. Isolates the inline-meta cost.
            if constexpr (!EnableEpllNoMeta) {
              const int topk = config.numExpertPerToken;
              *reinterpret_cast<index_t*>(stageBase) = FlatTokenIndex(config, myPe, srcTokId);
              float* mW = reinterpret_cast<float*>(stageBase + sizeof(index_t));
              index_t* mIdx = reinterpret_cast<index_t*>(stageBase + sizeof(index_t) +
                                                         (size_t)topk * sizeof(float));
              for (int e = 0; e < topk; ++e) {
                if (args.weightsBuf) mW[e] = args.weightsBuf[(size_t)srcTokId * topk + e];
                mIdx[e] = args.tokenIndices[(size_t)srcTokId * topk + e];
              }
            }
            uint32_t* flag = reinterpret_cast<uint32_t*>(stageBase + config.HiddenBytes(sizeof(T)));
            core::AtomicStoreReleaseSystem(flag, 1u);
          }
        }
      } else if constexpr (EnableLL128) {
        // [exp/epll-port] LL128: header warp writes the whole token into the interleaved staging
        // slot on the peer (combineInp@destPe), then a release flag right after the hidden data
        // ("flag next to data") so flag-visible => data-visible. One warp per token (no warp-group
        // split) keeps the data+flag ordering on a single wave. The receiver deflags locally below.
        if (inGroupWarpId == 0) {
          uint8_t* stageBase = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                               (size_t)destTokId * config.MaxXferBytesPerToken();
          core::WarpCopy<T, 2>(reinterpret_cast<T*>(stageBase), args.inpTokenBuf + srcTokOffset,
                               hiddenDim);
          if (laneId == 0) {
            // Whole-wave vmem writes drained, then publish the flag (release, system scope).
            __threadfence_system();
            uint32_t* flag = reinterpret_cast<uint32_t*>(stageBase + config.HiddenBytes(sizeof(T)));
            core::AtomicStoreReleaseSystem(flag, 1u);
          }
        }
      } else {
        core::WarpCopy<T, 2>(dispatchOut + destTokOffset + warpDimOffset,
                             args.inpTokenBuf + srcTokOffset + warpDimOffset, warpDimChunk);
      }

      // Deferred remote metadata scalar writes (header warp only), issued after the
      // big copy so the small xGMI transactions overlap the copy's store traffic.
      // [exp/epll-port] Step B: epll packs metadata inline (above) and unpacks on the receiver,
      // so it skips these per-field remote scalar writes; only the LOCAL dispDestTokIdMap is kept.
      if (inGroupWarpId == 0) {
        if (laneId == 0) {
          // SPSC already consumed destPeTokenCounter for the local slot index above.
          if constexpr (!EnableSPSC && !EnableEpll) atomicAdd(args.destPeTokenCounter + destPe, 1);
          args.dispDestTokIdMap[i] = FlatTokenIndex(config, destPe, destTokId);
          if constexpr (!EnableEpll || kEpllDeferMeta) {
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
                FlatTokenIndex(config, myPe, srcTokId);
          }
        }
        // Write weights and indices: only warp 0 writes
        if constexpr (!EnableEpll || kEpllDeferMeta) {
          if (laneId < config.numExpertPerToken) {
            if (args.weightsBuf) {
              args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
                  destPe)[destTokId * config.numExpertPerToken + laneId] =
                  args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
            }
            args.shmemOutIndicesMemObj->template GetAs<index_t*>(
                destPe)[destTokId * config.numExpertPerToken + laneId] =
                args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
          }
        }
      }

      // Write scales: split across 4 warps
      if (hasScales) {
        size_t scaleSize = config.scaleDim * config.scaleTypeSize;
        size_t scalePerWarp = (scaleSize + kWarpsPerGroup - 1) / kWarpsPerGroup;
        size_t myScaleOffset = (size_t)inGroupWarpId * scalePerWarp;
        size_t myScaleChunk =
            (myScaleOffset < scaleSize) ? min(scaleSize - myScaleOffset, scalePerWarp) : 0;

        size_t destScaleOffset = (size_t)destTokId * scaleSize;
        size_t srcScaleOffset = (size_t)srcTokId * scaleSize;
        core::WarpCopy(args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) +
                           destScaleOffset + myScaleOffset,
                       args.scalesBuf + srcScaleOffset + myScaleOffset, myScaleChunk);
      }
    }
  }

  if constexpr (EnableEpll) {
    // [exp/epll-port] epll completion: data is already in the final dispatchOut (direct write).
    // No grid barrier gating the count (counts arrived upfront), no warp0-serial handshake, no
    // deflag copy. Every block reads the upfront per-source counts -> prefix bases, then the whole
    // grid (per-thread grid-stride) WAITS on each slot's release flag so the kernel exits only
    // once all inbound tokens have landed (the GEMM-ready guarantee). A single cheap intra-device
    // grid barrier at the end resets the count signals for the next iteration's reuse.
    constexpr int kMaxLLPes = 64;
    __shared__ index_t llBase[kMaxLLPes + 1];
    __shared__ index_t llTotal;
    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    MORI_TRACE_NEXT(seq, Slot::DispatchWaitPeerToken);
    if (thdId == 0) {
      index_t acc = 0;
      for (int s = 0; s < npes; ++s) {
        index_t v;
        while ((v = core::AtomicLoadRelaxedSystem(recvTokenNums + s)) <= 0) {
          if constexpr (EnableSnooze) __builtin_amdgcn_s_sleep(1);
        }
        llBase[s] = acc;
        acc += (v - 1);
      }
      llBase[npes] = acc;
      llTotal = acc;
    }
    __syncthreads();

    index_t total = llTotal;
    size_t hiddenBytes = config.HiddenBytes(sizeof(T));
    size_t slotStride = config.MaxXferBytesPerToken();
    uint8_t* stageLocal = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>();
    if constexpr (EnableEpllLine) {
      // [exp/epll-port] epll_full completion: each WARP copies one slot's self-flagged LL128 lines
      // from local staging into the clean contiguous dispatchOut (poll-per-line, no fence, metadata
      // already landed via the deferred remote scalar writes). Per-warp grid-stride over all tokens.
      const size_t hiddenWords = hiddenBytes / sizeof(uint64_t);
      const size_t llStride = epll::LLStrideBytes(hiddenBytes);
      uint64_t* dispWords =
          reinterpret_cast<uint64_t*>(args.intraNodeTokBufs.dispatchOut->template GetAs<T*>());
      for (int r = globalWarpId; r < total; r += globalWarpNum) {
        int src = 0;
        while (src < npes && r >= llBase[src + 1]) ++src;
        index_t localIdx = r - llBase[src];
        index_t slot = (index_t)src * config.MaxNumTokensToRecvPerRank() + localIdx;
        const uint8_t* lineBase = stageLocal + (size_t)slot * llStride;
        uint64_t* dst = dispWords + (size_t)slot * hiddenWords;
        epll::LLReadToken(lineBase, dst, hiddenWords, epll::kLLReadyFlag, laneId, warpSize,
                          epll::kDefaultPollIters);
      }
    } else {
      int gThd = blockIdx.x * thdNum + thdId;
      int gThdNum = gridDim.x * thdNum;
      for (int r = gThd; r < total; r += gThdNum) {
        int src = 0;
        while (src < npes && r >= llBase[src + 1]) ++src;
        index_t localIdx = r - llBase[src];
        index_t slot = (index_t)src * config.MaxNumTokensToRecvPerRank() + localIdx;
        uint8_t* mBase = stageLocal + (size_t)slot * slotStride;
        uint32_t* flag = reinterpret_cast<uint32_t*>(mBase + hiddenBytes);
        while (core::AtomicLoadRelaxedSystem(flag) == 0u) {
          if constexpr (EnableSnooze) __builtin_amdgcn_s_sleep(1);
        }
        // [exp/epll-port] Step B: acquire fence pairs with the sender's release-flag store so the
        // inline meta-block (and hidden in dispatchOut) is visible, then unpack it LOCALLY into the
        // dense metadata buffers (replaces the sender's remote scalar writes with a local scatter).
        // [exp/epll-port] EnableEpllNoMeta ablation: the sender already wrote metadata via deferred
        // remote scalar stores (base-style), so there is nothing to unpack here; just reset the flag.
        __threadfence_system();
        if constexpr (!EnableEpllNoMeta) {
          const int topk = config.numExpertPerToken;
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[slot] =
              *reinterpret_cast<index_t*>(mBase);
          float* mW = reinterpret_cast<float*>(mBase + sizeof(index_t));
          index_t* mIdx =
              reinterpret_cast<index_t*>(mBase + sizeof(index_t) + (size_t)topk * sizeof(float));
          float* wOut = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>();
          index_t* idxOut = args.shmemOutIndicesMemObj->template GetAs<index_t*>();
          for (int e = 0; e < topk; ++e) {
            if (args.weightsBuf) wOut[(size_t)slot * topk + e] = mW[e];
            idxOut[(size_t)slot * topk + e] = mIdx[e];
          }
        }
        core::AtomicStoreRelaxedSystem(flag, 0u);  // reset for next-iter reuse
      }
    }

    __syncthreads();
    __threadfence_system();
    if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);
    if (blockIdx.x == 0 && thdId == 0) {
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      *args.totalRecvTokenNum = total;
      for (int s = 0; s < npes; ++s) {
        core::AtomicStoreRelaxedSystem(recvTokenNums + s, 0);
        args.destPeTokenCounter[s] = 0;
      }
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
  } else if constexpr (EnableLL128) {
    // [exp/epll-port] LL128 completion: no grid barrier, no warp0-serial count handshake.
    // Counts already arrived upfront; block 0 reads them, then all its warps poll per-token
    // flags in parallel and deflag (copy hidden data only) from the interleaved staging buffer
    // into the contiguous dispatchOut that the downstream (external) gemm reads. Staging is the
    // local combineInp (uncached, free during dispatch), so flag-visible => data-visible.
    constexpr int kMaxLLPes = 64;
    __shared__ index_t llBase[kMaxLLPes + 1];
    __shared__ index_t llTotal;
    MORI_TRACE_NEXT(seq, Slot::DispatchWaitPeerToken);
    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    if (blockIdx.x == 0) {
      if (thdId == 0) {
        index_t acc = 0;
        for (int s = 0; s < npes; ++s) {
          index_t v;
          while ((v = core::AtomicLoadRelaxedSystem(recvTokenNums + s)) <= 0) {
            if constexpr (EnableSnooze) __builtin_amdgcn_s_sleep(1);
          }
          llBase[s] = acc;
          acc += (v - 1);
        }
        llBase[npes] = acc;
        llTotal = acc;
      }
      __syncthreads();

      index_t total = llTotal;
      size_t hiddenBytes = config.HiddenBytes(sizeof(T));
      size_t slotStride = config.MaxXferBytesPerToken();
      uint8_t* stageLocal = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>();
      T* dispLocal = args.intraNodeTokBufs.dispatchOut->template GetAs<T*>();
      for (int r = warpId; r < total; r += warpNum) {
        int src = 0;
        while (src < npes && r >= llBase[src + 1]) ++src;
        index_t localIdx = r - llBase[src];
        index_t slot = (index_t)src * config.MaxNumTokensToRecvPerRank() + localIdx;
        uint8_t* slotBase = stageLocal + (size_t)slot * slotStride;
        uint32_t* flag = reinterpret_cast<uint32_t*>(slotBase + hiddenBytes);
        // All lanes spin on the (uncached) flag; once set, the producer's prior data writes are
        // in memory, so the deflag copy below reads valid data.
        while (core::AtomicLoadRelaxedSystem(flag) == 0u) {
          if constexpr (EnableSnooze) __builtin_amdgcn_s_sleep(1);
        }
        core::WarpCopy<T, 2>(dispLocal + (size_t)slot * hiddenDim,
                             reinterpret_cast<T*>(slotBase), hiddenDim);
        if (laneId == 0) core::AtomicStoreRelaxedSystem(flag, 0u);  // reset for next iter reuse
      }
      __syncthreads();

      if (thdId == 0) {
        *args.totalRecvTokenNum = total;
        for (int s = 0; s < npes; ++s) {
          core::AtomicStoreRelaxedSystem(recvTokenNums + s, 0);  // consume upfront count signal
          args.destPeTokenCounter[s] = 0;
        }
        args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
      }
    }
  } else {
    __syncthreads();
    if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);

    // Send token num & token to expert mapping to other ranks
    MORI_TRACE_NEXT(seq, Slot::DispatchNotifyPeer);
    if (globalWarpId == 0) {
      for (int destPe = laneId; destPe < npes; destPe += warpSize) {
        // Wait until all tokens are sent
        shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
        __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);

        // Add 1 so that when token number == 0, receiver side still know the signal is sent
        index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
        index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
        if constexpr (EnableSnooze) {
          // [exp/epll-port] P3: s_sleep backoff on the cross-device reuse spin (ep_ll snooze).
          while (core::AtomicLoadRelaxedSystem(signal) != 0) {
            __builtin_amdgcn_s_sleep(1);
          }
        } else {
          shmem::ShmemInt32WaitUntilEquals(signal, 0);
        }
        core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
      }
    }

    // Phase 2: recv token
    // Each warp wait until sender finished by waiting token number signal
    MORI_TRACE_NEXT(seq, Slot::DispatchWaitPeerToken);
    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    if (globalWarpId == 0) {
      for (int destPe = laneId; destPe < npes; destPe += warpSize) {
        index_t* signal = recvTokenNums + destPe;
        index_t recvTokenNum;
        if constexpr (EnableSnooze) {
          // [exp/epll-port] P3: s_sleep backoff on the cross-device count-wait spin (ep_ll snooze).
          index_t v;
          while ((v = core::AtomicLoadRelaxedSystem(signal)) <= 0) {
            __builtin_amdgcn_s_sleep(1);
          }
          recvTokenNum = v - 1;
        } else {
          recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
        }
        core::AtomicStoreRelaxedSystem(signal, 0);
        atomicAdd(args.totalRecvTokenNum, recvTokenNum);

        // reset local counter
        args.destPeTokenCounter[destPe] = 0;
      }

      // reset counter
      if (laneId == 0) {
        args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
      }
    }
  }

#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

template <typename T, bool EnableStdMoE = false>
__global__ void EpDispatchIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  EpDispatchIntraNodeKernel_body<T, EnableStdMoE>(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineIntraNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T, bool UseP2PRead = true, bool EnableStdMoE = false,
          bool UseFp8DirectCast = false, bool UseFp8BlockwiseQuant = false, bool UseWeights = true,
          int Vec8Top8BlockElems = 0, bool EnableEpllP1 = false>
__device__ __forceinline__ void EpCombineIntraNodeKernel_body(EpDispatchCombineArgs<T> args) {
  using TokT =
      std::conditional_t<UseFp8DirectCast || UseFp8BlockwiseQuant, core::CombineInternalFp8, T>;
  static_assert(!(UseFp8DirectCast && UseFp8BlockwiseQuant),
                "Fp8 direct cast and blockwise quant are mutually exclusive");
  static_assert((!UseFp8DirectCast && !UseFp8BlockwiseQuant) || std::is_same_v<T, hip_bfloat16>,
                "Fp8 combine quant currently only supports bf16 input");
  static_assert((Vec8Top8BlockElems & (Vec8Top8BlockElems - 1)) == 0,
                "Vec8Top8BlockElems must be 0 or a power of two");
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int globalThdNum = gridDim.x * warpNum * warpSize;

  int myPe = config.rank;
  int npes = config.worldSize;

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::CombineStageInput);

  const uint64_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];
  // Copy input to shmem registered buffer so that other GPUs can access directly
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  // When TokT != T (e.g. fp8 combine), staging layout uses TokT-sized tokens
  const size_t hiddenDim = config.HiddenDimSz();
  const size_t hiddenBytes = hiddenDim * sizeof(TokT);
  const size_t weightBytes =
      (UseWeights && args.weightsBuf != nullptr) ? config.numExpertPerToken * sizeof(float) : 0;
  const size_t scaleBytes =
      UseFp8BlockwiseQuant ? static_cast<size_t>(args.fp8BlockwiseCombineScaleDim) * sizeof(float)
                           : 0;
  const size_t combXferBytes = hiddenBytes + scaleBytes + weightBytes;

  if constexpr (EnableStdMoE) {
#ifdef ENABLE_STANDARD_MOE_ADAPT
    InvokeConvertCombineInput<T, UseP2PRead>(args, myPe);
#endif
  } else if constexpr (UseP2PRead) {
    if (args.config.useExternalInpBuffer) {
      for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
        if constexpr (UseFp8BlockwiseQuant) {
          core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
              args.intraNodeTokBufs.combineInp->template GetAs<TokT*>() + i * hiddenDim,
              args.shmemInpScalesMemObj->template GetAs<float*>() +
                  i * args.fp8BlockwiseCombineScaleDim,
              args.inpTokenBuf + i * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
        } else if constexpr (!std::is_same_v<T, TokT> &&
                             std::is_same_v<TokT, core::CombineInternalFp8>) {
          core::WarpCastBf16ToCombineInternalFp8<T>(
              args.intraNodeTokBufs.combineInp->template GetAs<TokT*>() + i * hiddenDim,
              args.inpTokenBuf + i * hiddenDim, hiddenDim, laneId);
        } else {
          core::WarpCopy(args.intraNodeTokBufs.combineInp->template GetAs<T*>() + i * hiddenDim,
                         args.inpTokenBuf + i * hiddenDim, hiddenDim);
        }
      }
    }
    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineCopyWeights);
      if (args.weightsBuf) {
        for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
          core::WarpCopy(
              args.shmemInpWeightsMemObj->template GetAs<float*>() + i * config.numExpertPerToken,
              args.weightsBuf + i * config.numExpertPerToken, config.numExpertPerToken);
        }
      }
    }
  } else {
#ifdef ENABLE_PROFILER
    for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
      index_t destTokId = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe)[tokenIdx];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
      uint8_t* destStagingPtr = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                                SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
      if constexpr (UseFp8BlockwiseQuant) {
        core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
            reinterpret_cast<core::CombineInternalFp8*>(destStagingPtr),
            reinterpret_cast<float*>(destStagingPtr + hiddenBytes),
            args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
      } else if constexpr (!std::is_same_v<T, TokT> &&
                           std::is_same_v<TokT, core::CombineInternalFp8>) {
        core::WarpCastBf16ToCombineInternalFp8<T>(reinterpret_cast<TokT*>(destStagingPtr),
                                                  args.inpTokenBuf + tokenIdx * hiddenDim,
                                                  hiddenDim, laneId);
      } else {
        core::WarpCopy(reinterpret_cast<T*>(destStagingPtr),
                       args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim);
      }
    }
    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineCopyWeights);
      if (args.weightsBuf) {
        for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
          index_t destTokId =
              args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe)[tokenIdx];
          index_t destPe = PeFromFlatTokenIndex(config, destTokId);
          index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
          uint8_t* destStagingPtr =
              args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
              SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
          core::WarpCopy(reinterpret_cast<float*>(destStagingPtr + hiddenBytes + scaleBytes),
                         args.weightsBuf + tokenIdx * config.numExpertPerToken,
                         config.numExpertPerToken);
        }
      }
    }
#else
    for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
      index_t destTokId = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe)[tokenIdx];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
      uint8_t* destStagingPtr = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                                SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
      if constexpr (UseFp8BlockwiseQuant) {
        core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
            reinterpret_cast<core::CombineInternalFp8*>(destStagingPtr),
            reinterpret_cast<float*>(destStagingPtr + hiddenBytes),
            args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
      } else if constexpr (!std::is_same_v<T, TokT> &&
                           std::is_same_v<TokT, core::CombineInternalFp8>) {
        core::WarpCastBf16ToCombineInternalFp8<T>(reinterpret_cast<TokT*>(destStagingPtr),
                                                  args.inpTokenBuf + tokenIdx * hiddenDim,
                                                  hiddenDim, laneId);
      } else {
        core::WarpCopy(reinterpret_cast<T*>(destStagingPtr),
                       args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim);
      }
      if constexpr (UseWeights) {
        if (args.weightsBuf) {
          core::WarpCopy(reinterpret_cast<float*>(destStagingPtr + hiddenBytes + scaleBytes),
                         args.weightsBuf + tokenIdx * config.numExpertPerToken,
                         config.numExpertPerToken);
        }
      }
      if constexpr (EnableEpllP1) {
        // [exp/epll-port] P1: publish per-record readiness after staging this token to
        // destPe (replaces the global CrossDeviceBarrier). Flag index == data slot offset,
        // so the producer/consumer mapping is identical to the staged data.
        if (laneId == 0) {
          __threadfence_system();
          uint32_t* readyFlag = args.combineReadyFlagMemObj->template GetAs<uint32_t*>(destPe) +
                                SendBufSlotOffset(config, myPe, destLocalTokId);
          core::AtomicStoreRelaxedSystem(readyFlag, static_cast<uint32_t>(crossDeviceBarrierFlag));
        }
      }
    }
#endif
  }

  // Make sure copy on all GPUs are finished
  MORI_TRACE_NEXT(seq, Slot::CombineBarrier);
  if constexpr (EnableEpllP1) {
    // [exp/epll-port] P1: no global barrier; per-record flags gate each read below.
    // Advance the epoch so this launch's flags are distinguishable from the next.
    __syncthreads();
    if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);
  } else {
    CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);
  }
  *args.totalRecvTokenNum = 0;
  if (args.curRankNumToken == 0) return;

  MORI_TRACE_NEXT(seq, Slot::CombineAccumSetup);
  extern __shared__ char sharedMem[];
  // Layout: [srcPtrs] [srcWeightsPtr if UseWeights] [srcScalePtrs if UseFp8BlockwiseQuant];
  // host-side combine_shared_mem() must use the same flags.
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = nullptr;
  if constexpr (UseWeights) {
    srcWeightsPtr = reinterpret_cast<float**>(sharedMem) + warpNum * config.numExpertPerToken +
                    warpId * config.numExpertPerToken;
  }
  float** srcScalePtrs = nullptr;
  if constexpr (UseFp8BlockwiseQuant) {
    constexpr int scalePtrArrayOffset = UseWeights ? 2 : 1;
    srcScalePtrs = reinterpret_cast<float**>(sharedMem) +
                   scalePtrArrayOffset * warpNum * config.numExpertPerToken +
                   warpId * config.numExpertPerToken;
  }

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

  assert(config.numExpertPerToken < warpSize);
  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    // Prepare data pointers on different GPUs
    MORI_TRACE_NEXT(seq, Slot::CombinePreparePtrs);
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);

      if (destPe < config.worldSize) {
        if constexpr (UseP2PRead) {
          index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
          srcPtrs[j] = args.intraNodeTokBufs.combineInp->template GetAs<TokT*>(destPe) +
                       destLocalTokId * hiddenDim + hiddenDimOffset;
          if constexpr (UseWeights) {
            srcWeightsPtr[j] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                               destLocalTokId * config.numExpertPerToken;
          }
          if constexpr (UseFp8BlockwiseQuant) {
            float* scalePtr = args.shmemInpScalesMemObj->template GetAs<float*>(destPe) +
                              destLocalTokId * args.fp8BlockwiseCombineScaleDim;
            srcScalePtrs[j] = (scalePtr[0] < 0.0f) ? scalePtr : nullptr;
          }
        } else {
          srcPtrs[j] = reinterpret_cast<TokT*>(
                           args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                           SendBufSlotOffset(config, destPe, tokenId) * combXferBytes) +
                       hiddenDimOffset;
          if constexpr (UseWeights) {
            srcWeightsPtr[j] = reinterpret_cast<float*>(
                args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                SendBufSlotOffset(config, destPe, tokenId) * combXferBytes + hiddenBytes +
                scaleBytes);
          }
          if constexpr (UseFp8BlockwiseQuant) {
            float* scalePtr = reinterpret_cast<float*>(
                args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                SendBufSlotOffset(config, destPe, tokenId) * combXferBytes + hiddenBytes);
            srcScalePtrs[j] = (scalePtr[0] < 0.0f) ? scalePtr : nullptr;
          }
          if constexpr (EnableEpllP1) {
            // [exp/epll-port] P1: wait for this record's readiness flag (set by producer
            // destPe after it staged the token) instead of a global barrier.
            uint32_t* readyFlag = args.combineReadyFlagMemObj->template GetAs<uint32_t*>(myPe) +
                                  SendBufSlotOffset(config, destPe, tokenId);
            while (core::AtomicLoadRelaxedSystem(readyFlag) <
                   static_cast<uint32_t>(crossDeviceBarrierFlag)) {
              __builtin_amdgcn_s_sleep(1);
            }
            __threadfence_system();
          }
        }
      } else {
        srcPtrs[j] = nullptr;
        if constexpr (UseWeights) {
          srcWeightsPtr[j] = nullptr;
        }
        if constexpr (UseFp8BlockwiseQuant) {
          srcScalePtrs[j] = nullptr;
        }
      }
    }

    T* outPtr = args.intraNodeTokBufs.combineOut->template GetAs<T*>() + tokenId * hiddenDim +
                hiddenDimOffset;

    int validAccumCount = config.numExpertPerToken;
    if (config.worldSize <= 4) {
      {
        int isValid = 0;
        TokT* myTokPtr = nullptr;
        float* myScalePtr = nullptr;
        if (laneId < config.numExpertPerToken) {
          myTokPtr = srcPtrs[laneId];
          if constexpr (UseFp8BlockwiseQuant) {
            myScalePtr = srcScalePtrs[laneId];
          }
          isValid = (myTokPtr != nullptr) ? 1 : 0;
        }
        unsigned long long validMask = __ballot(isValid);
        validAccumCount = __popcll(validMask);
        if (validAccumCount < config.numExpertPerToken && isValid) {
          int myPos = __popcll(validMask & ((1ULL << laneId) - 1));
          srcPtrs[myPos] = myTokPtr;
          if constexpr (UseFp8BlockwiseQuant) {
            srcScalePtrs[myPos] = myScalePtr;
          }
        }
      }
    }

    if constexpr (UseFp8BlockwiseQuant) {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      if constexpr (Vec8Top8BlockElems != 0) {
        if (mwIter.warpsPerItem == 1) {
          core::WarpAccumFp8DequantFullBlockVec8Top8<T, core::CombineInternalFp8,
                                                     Vec8Top8BlockElems>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDim);
        } else if ((hiddenDimOffset & 0x7) == 0 && (hiddenDimSize & 0x7) == 0) {
          core::WarpAccumFp8DequantSegmentBlockVec8Top8<T, core::CombineInternalFp8,
                                                        Vec8Top8BlockElems>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDimOffset, hiddenDimSize);
        } else {
          // Misaligned segment: vec8 helper would fault on the load. Tiny scalar fallback.
          core::WarpAccumFp8DequantSegmentScalarTop8<T, core::CombineInternalFp8,
                                                     Vec8Top8BlockElems>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDimOffset, hiddenDimSize);
        }
      } else {
        if (mwIter.warpsPerItem == 1) {
          core::WarpAccumFp8DequantFull<T, core::CombineInternalFp8>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), validAccumCount, hiddenDim,
              args.fp8BlockwiseCombineScaleDim);
        } else {
          core::WarpAccumFp8DequantSegment<T, core::CombineInternalFp8>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), validAccumCount, hiddenDimOffset,
              hiddenDimSize, hiddenDim, args.fp8BlockwiseCombineScaleDim);
        }
      }
    } else if constexpr (!std::is_same_v<T, TokT> &&
                         std::is_same_v<TokT, core::CombineInternalFp8>) {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      core::WarpAccumCombineInternalFp8ToBf16(outPtr, reinterpret_cast<const TokT* const*>(srcPtrs),
                                              validAccumCount, laneId, hiddenDimSize);
    } else {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      // BENCH_NOOP_COMBINE: transport-only mirror of ep_ll (no topk sum)
      TokT* firstSrc = nullptr;
      for (int j = 0; j < validAccumCount; ++j) {
        if (srcPtrs[j] != nullptr) {
          firstSrc = srcPtrs[j];
          break;
        }
      }
      if (firstSrc != nullptr) {
        core::WarpCopy(outPtr, firstSrc, hiddenDimSize);
      }
    }

    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineAccumWeights);
      if (args.weightsBuf && inTokenPartId == mwIter.warpsPerItem - 1) {
        core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                      tokenId * config.numExpertPerToken,
                                  srcWeightsPtr, nullptr, config.numExpertPerToken,
                                  config.numExpertPerToken);
      }
    }
  }
}

template <typename T, bool UseP2PRead = true, bool EnableStdMoE = false,
          bool UseFp8DirectCast = false, bool UseFp8BlockwiseQuant = false, bool UseWeights = true,
          int Vec8Top8BlockElems = 0, bool EnableEpllP1 = false>
__global__ void EpCombineIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  EpCombineIntraNodeKernel_body<T, UseP2PRead, EnableStdMoE, UseFp8DirectCast, UseFp8BlockwiseQuant,
                                UseWeights, Vec8Top8BlockElems, EnableEpllP1>(args);
}

}  // namespace moe
}  // namespace mori
