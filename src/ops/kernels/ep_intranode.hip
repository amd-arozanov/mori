// KernelType::IntraNode — dispatch + combine for intra-node (P2P) mode.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

// Dispatch
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, _stdmoe, true)

// Dispatch Low Lantency (SMALL TOKEN)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeLLKernel, _stdmoe, true)

// [exp/epll-port] P3 (s_sleep snooze on cross-device signal spins) — bf16 only.
extern "C" __global__ void EpDispatchIntraNodeLLKernel_bf16_snooze(
    EpDispatchCombineArgs<hip_bfloat16> args) {
  EpDispatchIntraNodeLLKernel_body<hip_bfloat16, /*EnableStdMoE=*/false, /*EnableSnooze=*/true>(
      args);
}

// [exp/epll-port] P2 (SPSC deterministic source-owned slot, no remote atomicAdd) — bf16 only.
extern "C" __global__ void EpDispatchIntraNodeLLKernel_bf16_spsc(
    EpDispatchCombineArgs<hip_bfloat16> args) {
  EpDispatchIntraNodeLLKernel_body<hip_bfloat16, /*EnableStdMoE=*/false, /*EnableSnooze=*/false,
                                   /*EnableSPSC=*/true>(args);
}

// [exp/epll-port] LL128 (upfront counts + per-token flag-with-data + local deflag, no grid
// barrier / no warp0-serial count wait) — builds on SPSC; bf16 only.
extern "C" __global__ void EpDispatchIntraNodeLLKernel_bf16_ll128(
    EpDispatchCombineArgs<hip_bfloat16> args) {
  EpDispatchIntraNodeLLKernel_body<hip_bfloat16, /*EnableStdMoE=*/false, /*EnableSnooze=*/false,
                                   /*EnableSPSC=*/true, /*EnableLL128=*/true>(args);
}

// [exp/epll-port] epll split-dispatch (SPSC slot + upfront counts + DIRECT write to dispatchOut
// + per-slot release flag + multi-block parallel flag-wait; no grid barrier, no warp0-serial
// handshake, no staging/deflag) — bf16 only. Pairs with EpCombineIntraNodeKernel_bf16_nop2p_p1.
extern "C" __global__ void EpDispatchIntraNodeLLKernel_bf16_epll(
    EpDispatchCombineArgs<hip_bfloat16> args) {
  EpDispatchIntraNodeLLKernel_body<hip_bfloat16, /*EnableStdMoE=*/false, /*EnableSnooze=*/true,
                                   /*EnableSPSC=*/false, /*EnableLL128=*/false,
                                   /*EnableEpll=*/true>(args);
}

// [exp/epll-port] epll WITHOUT metadata-in-line: same direct-write + per-slot release flag, but
// metadata goes via deferred remote scalar writes (base-style) instead of the inline pack/unpack.
// Isolates the inline-meta serialization cost in dispatch copy_token. bf16 only.
extern "C" __global__ void EpDispatchIntraNodeLLKernel_bf16_epll_nometa(
    EpDispatchCombineArgs<hip_bfloat16> args) {
  EpDispatchIntraNodeLLKernel_body<hip_bfloat16, /*EnableStdMoE=*/false, /*EnableSnooze=*/true,
                                   /*EnableSPSC=*/false, /*EnableLL128=*/false,
                                   /*EnableEpll=*/true, /*EnableEpllNoMeta=*/true>(args);
}

// [exp/epll-port] epll_full: FULL ep_ll alignment. Self-flagged LL128 line transport (7 data words
// + 1 seq flag per 64B line), no per-token __threadfence_system, no group-epoch sync; receiver
// copies per-line from staging into dispatchOut. Metadata via deferred remote scalar writes. bf16.
extern "C" __global__ void EpDispatchIntraNodeLLKernel_bf16_epll_full(
    EpDispatchCombineArgs<hip_bfloat16> args) {
  EpDispatchIntraNodeLLKernel_body<hip_bfloat16, /*EnableStdMoE=*/false, /*EnableSnooze=*/true,
                                   /*EnableSPSC=*/false, /*EnableLL128=*/false,
                                   /*EnableEpll=*/true, /*EnableEpllNoMeta=*/false,
                                   /*EnableEpllLine=*/true>(args);
}

// Combine
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p, true, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _nop2p, false, false, false)

// [exp/epll-port] P1 (per-record readiness flags, replaces CrossDeviceBarrier) — bf16 nop2p.
extern "C" __global__ void EpCombineIntraNodeKernel_bf16_nop2p_p1(
    EpDispatchCombineArgs<hip_bfloat16> args) {
  EpCombineIntraNodeKernel_body<hip_bfloat16, /*UseP2PRead=*/false, /*EnableStdMoE=*/false,
                                /*UseFp8DirectCast=*/false, /*UseFp8BlockwiseQuant=*/false,
                                /*UseWeights=*/true, /*Vec8Top8BlockElems=*/0,
                                /*EnableEpllP1=*/true>(args);
}
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p_stdmoe, true, true, false)
WRAP_BOOL3(EpCombineIntraNodeKernel, bf16_nop2p_fp8cast, hip_bfloat16, false, false, true)
MORI_FP8_ANY(WRAP_BOOL4(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq, hip_bfloat16, false,
                        false, false, true))
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq_noweight_block128_vec8,
                        hip_bfloat16, false, false, false, true, false, 128))
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq_noweight_block256_vec8,
                        hip_bfloat16, false, false, false, true, false, 256))

// Convert (used with ENABLE_STANDARD_MOE_ADAPT)
extern "C" __global__ void mori_ConvertDispatchOutputKernel(ConvertDispatchOutputArgs args) {
  ConvertDispatchOutputDevice(args);
}
#define WRAP_CONVERT_COMBINE(suffix, T)                                                         \
  extern "C" __global__ void ConvertCombineInputKernel_##suffix(ConvertCombineInputArgs args) { \
    ConvertCombineInputDevice<T, true>(args);                                                   \
  }
WRAP_CONVERT_COMBINE(f32, float)
WRAP_CONVERT_COMBINE(bf16, hip_bfloat16)
MORI_FP8_OCP(WRAP_CONVERT_COMBINE(fp8_ocp, __hip_fp8_e4m3))
MORI_FP8_FNUZ(WRAP_CONVERT_COMBINE(fp8_fnuz, __hip_fp8_e4m3_fnuz))
