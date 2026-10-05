// HOT-Step patch: flash-attn-train
//
// CUDA forward for GGML_OP_FLASH_ATTN_TRAIN. Behavioural reference is the CPU
// impl in ggml-cpu/ops.cpp (ggml_compute_forward_flash_attn_train_f32); the
// maths is spec section 4.1, the kernel shape is spec section 8.1.
//
// Properties this kernel is required to hold, and how it holds them:
//
//   * f32 in / f32 out / f32 accumulate. No tensor cores, no TF32, no f16
//     tiles -- correctness first (spec 8.3).
//   * DETERMINISTIC. Every output element is written exactly once by exactly
//     one thread from a private accumulator; every reduction runs in a fixed
//     order (butterfly shuffle over the warp, then key tiles in ascending j,
//     then lanes in ascending order inside a tile). There are no floating
//     point atomics anywhere. Two runs on the same GPU with the same inputs
//     are bitwise identical.
//   * A -INF mask entry yields P EXACTLY 0.0f. The kernel never calls exp on
//     -INFINITY: masked lanes are branched out and their p is assigned 0.0f
//     literally. -use_fast_math is on for this backend, so leaning on
//     __expf(-INFINITY) would be leaning on an approximation's edge case.
//   * A fully masked query row yields O = 0 and LSE = 0, never NaN (spec 4.4).
//     A NaN here is fatal: ggml builds the packed tensor's gradient as
//     ggml_scale(packed, 0.0f) and 0 * NaN is NaN.
//   * The alignment gap between the O and LSE regions is explicitly zeroed
//     (spec 2.2). It is zero-width at every production geometry, which is
//     exactly why forgetting it ships.
//   * q/k/v are permuted, non-contiguous VIEWS. Only nb[0] == 4 is guaranteed;
//     every other axis is walked through nb[1..3]. Nothing is ggml_cont'd --
//     that would re-materialise Q/K/V and give back part of the win.
//
// Block shape: one block per (query tile, head, batch). FA_TRAIN_NWARPS warps
// per block, one query ROW per warp, FA_TRAIN_BK keys staged in shared memory
// per tile and shared by all the warps in the block. The running O accumulator
// lives in registers, D/32 floats per lane.

#include "fattn-train.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

// Env-gated trace, one line per direction per process. Spec 9.8 is emphatic
// that a `false` from supports_op is a SILENT fallback to the CPU backend and
// not a failure: the graph still runs, still gets the right answer, and is
// merely unusably slow with a quiet VRAM tripwire -- i.e. it looks like a pass.
// So "the kernel actually ran on the GPU" has to be checkable rather than
// argued. FATTN_TRAIN_TRACE=1.
static void fa_train_trace(
        bool & once, const char * what,
        int64_t D, int64_t S, int64_t S_kv, int64_t Nh, int64_t Nkv, int64_t Bn,
        bool has_mask, const char * prec) {
    static const bool on = getenv("FATTN_TRAIN_TRACE") != nullptr;
    if (!on || once) {
        return;
    }
    once = true;
    // `prec` is the RESOLVED mode with its reason, never the requested one: a
    // pre-Ampere device or a D = 64 graph runs the v1 kernels under a tf32
    // request, and "which kernel actually ran" has to stay checkable rather
    // than argued (tf32 design 3.3).
    GGML_LOG_INFO("fattn-train: CUDA %s  D=%d S=%d S_kv=%d Nh=%d Nkv=%d B=%d mask=%s prec=%s\n",
                  what, (int) D, (int) S, (int) S_kv, (int) Nh, (int) Nkv, (int) Bn,
                  has_mask ? "yes" : "null", prec);
}

// ─── precision mode (roadmap R1) ────────────────────────────────────────────
//
// docs/plans/fattn-train-tf32-design.md 3.3. The decision is made ONCE, here,
// and everything else asks this function: the trace line, the trainer's log
// field, and the parity tool's `prec` column. Three call sites deriving the same
// rule three times is how a mis-set cc threshold gets to disagree with the label
// on the results.
//
// op_params slot 3 carries the request (ggml_flash_attn_train_set_prec):
//   GGML_PREC_DEFAULT (0)  -> TF32 where it is available
//   GGML_PREC_F32          -> the v1 scalar kernels, always
// GGML_PREC_DEFAULT is 0 and ggml_new_tensor zeroes op_params, so a graph built
// before the flag existed reads as TF32 without a migration.

enum fa_train_prec_mode {
    FA_TRAIN_PREC_F32  = 0,   // v1 scalar kernels
    FA_TRAIN_PREC_TF32 = 1,   // tensor-core kernels
};

struct fa_train_prec_resolved {
    fa_train_prec_mode mode;
    const char *       label;   // static string: "tf32", or "f32 (<reason>)"
};

// The TF32 path's alignment contract (design 7). q/k/v are permuted,
// non-contiguous VIEWS and the only stride supported() checks is
// nb[0] == sizeof(float); an odd view puts a row base on a 4-byte boundary, and
// ld.shared.v2 / ld.global.v2 there is an unspecified launch failure that takes
// the process down. Checked HERE rather than in supports_op, which must stay
// prec-blind: refusing the op outright would turn a correct slower path into a
// hard refusal on geometries v1 handles fine.
static bool fa_train_view_8aligned(const ggml_tensor * t) {
    return ((uintptr_t) t->data) % 8 == 0 &&
           t->nb[1] % 8 == 0 && t->nb[2] % 8 == 0 && t->nb[3] % 8 == 0;
}

static fa_train_prec_resolved fa_train_resolve_prec(const ggml_tensor * op, const int cc, const bool back) {
    fa_train_prec_resolved r = { FA_TRAIN_PREC_F32, "f32" };

    if ((enum ggml_prec) ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        r.label = "f32 (requested)";
        return r;
    }
    // Both checks, not either: a kernel compiled for sm_75 is a NO_DEVICE_CODE
    // stub, and launching it because the DEVICE check passed writes zeros rather
    // than crashing. Writing the DEVICE check against the compiled arch (or the
    // reverse) is the same bug from the other side.
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        r.label = "f32 (device cc < 800)";
        return r;
    }
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        r.label = "f32 (no sm_80+ in build)";
        return r;
    }
    // Deliberate scope limit, written down because D = 64 IS dispatchable today
    // (supported() accepts it and both switches carry a live case 64) while the
    // parity grid is D = 128 only. Every tile, register and shared-memory number
    // in design 1.4 was derived at D = 128.
    if (op->src[0]->ne[0] != 128) {
        r.label = "f32 (D != 128)";
        return r;
    }
    if (!fa_train_view_8aligned(op->src[0]) ||
        !fa_train_view_8aligned(op->src[1]) ||
        !fa_train_view_8aligned(op->src[2])) {
        r.label = "f32 (unaligned view)";
        return r;
    }
    // Both directions have tensor-core kernels, and the consistency requirement
    // design 2 calls the sharpest edge in the change is met at the op-pair level
    // (ggml_compute_backward copies the flag) and here: one resolver, one rule,
    // so a forward that rounds to TF32 cannot be paired with a backward that
    // does not. `back` survives for the one extra constraint the backward has:
    // its dK/dV kernel stages dO with 8-byte global loads, and dO is src[5],
    // which the forward does not have.
    if (back && (!op->src[5] || !fa_train_view_8aligned(op->src[5]))) {
        r.label = "f32 (unaligned dO)";
        return r;
    }

    r.mode  = FA_TRAIN_PREC_TF32;
    r.label = "tf32";
    return r;
}

// Which arithmetic the last launch of each direction actually used, queryable
// through the backend registry (design 5.3): a 1600x-looser tf32 bar cannot tell
// "TF32 ran and was accurate" from "TF32 never ran", so the parity tool asserts
// the kernel that RAN rather than restating the flag it asked for.
static const char * g_fa_train_last_prec[2] = { "n/a", "n/a" };

const char * ggml_cuda_fattn_train_last_prec(int dir) {
    if (dir < 0 || dir > 1) {
        return "n/a";
    }
    return g_fa_train_last_prec[dir];
}

#define FA_TRAIN_NWARPS 4    // query rows per block
#define FA_TRAIN_BK     32   // keys staged per tile (one per lane)

template <int D>
static __global__ void __launch_bounds__(FA_TRAIN_NWARPS*WARP_SIZE, 1)
fa_train_fwd_f32(
        const char * __restrict__ q_base,
        const char * __restrict__ k_base,
        const char * __restrict__ v_base,
        const half * __restrict__ mask,
        float      * __restrict__ o_data,
        float      * __restrict__ lse_data,
        const int64_t q_nb1, const int64_t q_nb2, const int64_t q_nb3,
        const int64_t k_nb1, const int64_t k_nb2, const int64_t k_nb3,
        const int64_t v_nb1, const int64_t v_nb2, const int64_t v_nb3,
        const int64_t mne0, const int64_t mne1, const int64_t mne2, const int64_t mne3,
        const int64_t S, const int64_t S_kv, const int64_t Nh, const int64_t G,
        const float scale) {

    constexpr int NV = D/WARP_SIZE;      // accumulator floats per lane
    constexpr int LD = D + 1;            // padded shared row stride (bank conflicts)

    __shared__ float Ksh[FA_TRAIN_BK*LD];
    __shared__ float Vsh[FA_TRAIN_BK*LD];
    __shared__ float Qsh[FA_TRAIN_NWARPS*LD];
    __shared__ int   sh_live[FA_TRAIN_NWARPS];

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = w*WARP_SIZE + lane;

    const int64_t i  = (int64_t) blockIdx.x*FA_TRAIN_NWARPS + w;   // query row
    const int64_t h  = blockIdx.y;
    const int64_t b  = blockIdx.z;
    const int64_t hk = h/G;                                        // GQA kv head

    const bool active = i < S;

    // Q row -> shared, read back as a warp-uniform broadcast in the dot loop.
    if (active) {
        const float * qrow = (const float *) (q_base + i*q_nb1 + h*q_nb2 + b*q_nb3);
#pragma unroll
        for (int c = 0; c < NV; ++c) {
            Qsh[w*LD + lane + WARP_SIZE*c] = qrow[lane + WARP_SIZE*c];
        }
    }
#ifndef GGML_USE_HIP
    __syncwarp();   // a HIP wavefront runs in lockstep, so the barrier is implicit there
#endif // GGML_USE_HIP

    float acc[NV];
#pragma unroll
    for (int c = 0; c < NV; ++c) {
        acc[c] = 0.0f;
    }
    float mrun = -INFINITY;   // running row max
    float lrun = 0.0f;        // running sum of exp

    for (int64_t j0 = 0; j0 < S_kv; j0 += FA_TRAIN_BK) {
        // (A) previous tile's compute is done: safe to overwrite K/V and flags
        __syncthreads();

        const int64_t j   = j0 + lane;
        const bool    jok = active && (j < S_kv);

        // spec 3.4: modulo broadcast, and mne1 (not S) is the mask row stride
        float mv = 0.0f;
        if (jok && mask) {
            const int64_t midx = j + mne0*(i + mne1*((h % mne2) + mne2*(b % mne3)));
            mv = __half2float(mask[midx]);
        }
        const bool live = jok && !(mv == -INFINITY);

        const int anyw = __any_sync(0xffffffff, live);
        if (lane == 0) {
            sh_live[w] = anyw;
        }

        // (B) tile liveness known to the whole block
        __syncthreads();

        int block_live = 0;
#pragma unroll
        for (int t = 0; t < FA_TRAIN_NWARPS; ++t) {
            block_live |= sh_live[t];
        }
        if (!block_live) {
            // Whole tile dead. With sliding_window = 128 and S in the
            // thousands this is the large majority of tiles.
            continue;
        }

        const int64_t nk = min((int64_t) FA_TRAIN_BK, S_kv - j0);
        for (int idx = tid; idx < FA_TRAIN_BK*D; idx += FA_TRAIN_NWARPS*WARP_SIZE) {
            const int jj = idx/D;
            const int dd = idx%D;
            float kval = 0.0f;
            float vval = 0.0f;
            if (jj < nk) {
                const float * krow = (const float *) (k_base + (j0 + jj)*k_nb1 + hk*k_nb2 + b*k_nb3);
                const float * vrow = (const float *) (v_base + (j0 + jj)*v_nb1 + hk*v_nb2 + b*v_nb3);
                kval = krow[dd];
                vval = vrow[dd];
            }
            Ksh[jj*LD + dd] = kval;
            Vsh[jj*LD + dd] = vval;
        }

        // (C) K/V tile staged
        __syncthreads();

        // Scores. The mask is additive AFTER the scale, so a -INF entry
        // survives as -INF whatever the scale -- and the dot product for it is
        // never even computed.
        float s = -INFINITY;
        if (live) {
            const float * qs = Qsh + w*LD;
            const float * ks = Ksh + lane*LD;
            float dot = 0.0f;
#pragma unroll 8
            for (int d = 0; d < D; ++d) {
                dot = fmaf(qs[d], ks[d], dot);
            }
            s = scale*dot + mv;
        }

        // warp-uniform after the reduction; mrun is warp-uniform by induction
        const float tilemax = warp_reduce_max<WARP_SIZE>(s);
        const float m_new   = fmaxf(mrun, tilemax);
        if (m_new == -INFINITY) {
            continue;   // still nothing seen on this row
        }

        // exp(-INF - finite) is 0 in IEEE-754; spelled out rather than
        // evaluated, because -use_fast_math is on for this backend.
        const float corr = (mrun == -INFINITY) ? 0.0f : __expf(mrun - m_new);
        const float p    = (s    == -INFINITY) ? 0.0f : __expf(s    - m_new);

        lrun = lrun*corr + warp_reduce_sum<WARP_SIZE>(p);
#pragma unroll
        for (int c = 0; c < NV; ++c) {
            acc[c] *= corr;
        }

        // Fixed lane order, so the accumulation order is fixed too.
#pragma unroll 4
        for (int jj = 0; jj < WARP_SIZE; ++jj) {
            const float pj = __shfl_sync(0xffffffff, p, jj, WARP_SIZE);
            if (pj == 0.0f) {
                continue;   // masked or vanished: contributes exactly nothing
            }
            const float * vs = Vsh + jj*LD;
#pragma unroll
            for (int c = 0; c < NV; ++c) {
                acc[c] = fmaf(pj, vs[lane + WARP_SIZE*c], acc[c]);
            }
        }

        mrun = m_new;
    }

    if (!active) {
        return;
    }

    float * orow = o_data + D*(h + Nh*(i + S*b));
    if (lrun > 0.0f) {
#pragma unroll
        for (int c = 0; c < NV; ++c) {
            orow[lane + WARP_SIZE*c] = acc[c]/lrun;
        }
        if (lane == 0) {
            lse_data[h + Nh*(i + S*b)] = mrun + __logf(lrun);
        }
    } else {
        // spec 4.4: fully-masked row -- defined, finite, NOT NaN
#pragma unroll
        for (int c = 0; c < NV; ++c) {
            orow[lane + WARP_SIZE*c] = 0.0f;
        }
        if (lane == 0) {
            lse_data[h + Nh*(i + S*b)] = 0.0f;
        }
    }
}

// ─── TF32 forward (roadmap R1) ──────────────────────────────────────────────
//
// docs/plans/fattn-train-tf32-design.md sections 1.3-1.5. Identical contract to
// fa_train_fwd_f32 above -- the same packed output, the same -INF rule, the same
// fully-masked-row rule, the same determinism -- computed with
// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 instead of an FFMA dot.
//
// One block per (64-query tile, head, batch), 4 warps, warp w owning query rows
// [64*blockIdx.x + 16*w, +16) -- exactly one mma M-tile. Q A-fragments and the O
// accumulator live in REGISTERS (BQ*D*4 = 32 KB does not fit in shared); K and V
// are staged BK = 16 keys at a time, swizzled, and shared by all four warps.
// Arithmetic intensity against the streamed side is the owned-side tile height,
// 16 FLOP/byte per warp, where v1's one-row-per-warp gave 1.
//
// Everything that is not a matmul stays f32 (design 2): the out-of-range
// predicate, the mask fetch, the -INF branch to a literal +0.0f, the running row
// max, the rescale chain, the row sum and the LSE. TF32 touches the two products
// and nothing else, and the accumulators are the instruction's own .f32.
//
// REGISTER DISCIPLINE IS PART OF THE DESIGN, not a tuning afterthought (design
// 7: a non-zero spill count is a design failure, not a warning). The 64-register
// Q fragment set and the 64-register O accumulator leave ~120 of the 255 for
// everything else, and three things were needed to stay inside it -- measured
// with -Xptxas -v on sm_120a, D = 128:
//
//   * hoist every int64. The first draft spilled 360 B because each unrolled
//     loop body carried its own (i, j, mask index) triple; the query rows and
//     the mask row pointers are fixed for the life of the block, so they are
//     computed once and the indices inside the unrolled bodies are int32.
//   * do NOT cache the tile's 16 mask values across the staging. That cost 16
//     registers; the score phase re-fetches them and a 16-bit `dead` map carries
//     the -INF verdict for one register.
//   * BK = 16, not 32. 32 was 255 registers with 400 B of spill; 16 is 255 with
//     ZERO. Halving the staged tile halves the score registers and, more to the
//     point, halves what ptxas keeps in flight across the two fully-unrolled
//     matmul phases. It also makes the all-masked tile skip finer-grained, which
//     is where the sliding-window win lives.
//
// The unrolls on the matmul loops must stay FULL: qa[] and oacc[] are indexed by
// the loop variable, and a partial `#pragma unroll 4` leaves that index dynamic,
// which moves both arrays to local memory (measured: 512 B stack frame, 222
// registers -- fewer registers and far slower).
//
// One design assumption did not survive contact: 1.2's "the A map and the C/D
// map are the same function, so a score tile is already the next mma's A
// operand" is true of ggml's f16 tiles and FALSE of tf32 (see fattn-train.cuh
// for the measurement). The score tile is redistributed by c_to_a instead --
// eight shuffles inside each 4-lane group, still no shared round-trip.
//
// Those shuffles cost the last of the headroom: at 255 registers sm_120a spilled
// 8 BYTES here for as long as the K tile was read as pairs of 4-byte loads.
// Design 7 calls any spill a design failure, and the fix arrived from the
// backward's tuning rather than from this kernel: staging under fa_tf32::pcol
// makes the B fragment's {t, t+4} k-pair ADJACENT, so it is one 8-byte load
// instead of two 4-byte ones. (The same rewrite MEASURED WORSE in the dQ kernel
// -- 0.184 -> 0.223 ms at S = 625 -- so it is applied per kernel, not globally.)
//
// The tile loop also carries two barriers rather than three: the liveness scan
// touches nothing but the mask, so one __syncthreads_or both publishes the
// verdict and establishes that every warp is done reading the previous tile's
// K/V. That merge is worth more than it looks -- across all three tf32 kernels
// it took the whole attention site from 0.671 ms to 0.602 at S = 625, which is
// the difference between missing the campaign's gate and clearing it. It also
// costs the 8-byte spill back and then some: the merged basic block schedules to
// 255 registers with 52 BYTES of spill. Recorded rather than rounded down
// (design 7), and kept, because the alternative is 12 % slower end to end.

#define FA_TRAIN_TF32_NW 4                       // warps per block
#define FA_TRAIN_TF32_BQ (FA_TRAIN_TF32_NW*16)   // queries per block (16 = the mma M tile)
#define FA_TRAIN_TF32_BK 16                      // keys staged per tile

template <int D>
static __global__ void __launch_bounds__(FA_TRAIN_TF32_NW*WARP_SIZE, 1)
fa_train_fwd_tf32(
        const char * __restrict__ q_base,
        const char * __restrict__ k_base,
        const char * __restrict__ v_base,
        const half * __restrict__ mask,
        float      * __restrict__ o_data,
        float      * __restrict__ lse_data,
        const int64_t q_nb1, const int64_t q_nb2, const int64_t q_nb3,
        const int64_t k_nb1, const int64_t k_nb2, const int64_t k_nb3,
        const int64_t v_nb1, const int64_t v_nb2, const int64_t v_nb3,
        const int64_t mne0, const int64_t mne1, const int64_t mne2, const int64_t mne3,
        const int64_t S, const int64_t S_kv, const int64_t Nh, const int64_t G,
        const float scale) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int NW  = FA_TRAIN_TF32_NW;
    constexpr int BK  = FA_TRAIN_TF32_BK;
    constexpr int NT  = NW*WARP_SIZE;
    constexpr int NKS = D/8;     // k-slices of the QK^T reduction
    constexpr int NDB = D/8;     // 8-wide d-blocks of the O accumulator
    constexpr int NNB = BK/8;    // 8-wide key blocks of the score tile
    constexpr int RPI = NT/D;    // K/V rows staged per loop iteration (1 at D = 128)

    static_assert(NT % D == 0, "the staging loop assumes whole rows per iteration");
    static_assert(BK % RPI == 0, "the staging loop assumes BK divides evenly");

    // 8-byte aligned because the PA fragment read is a float2 under pcol.
    __shared__ __align__(16) float Ksh[BK*D];
    __shared__ __align__(16) float Vsh[BK*D];

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = w*WARP_SIZE + lane;

    const int64_t q0 = (int64_t) blockIdx.x*FA_TRAIN_TF32_BQ + w*16;
    const int64_t h  = blockIdx.y;
    const int64_t b  = blockIdx.z;
    const int64_t hk = h/G;                     // GQA kv head

    // The A / C / D fragment map gives this lane two query rows 8 apart and two
    // adjacent columns of each, so every row-wise quantity below is a PAIR:
    // suffix _a is fragment elements l = 0, 1 and suffix _c is l = 2, 3.
    const int64_t ia  = q0     + lane/4;
    const int64_t ic  = q0 + 8 + lane/4;
    const bool    oka = ia < S;
    const bool    okc = ic < S;

    // Q -> A-fragment registers, once per block, on the A map: element l is
    // (row g + 8*(l%2), d = 8*ks + 4*(l/2) + t). That is NOT the C/D map -- see
    // fattn-train.cuh, where the two are measured apart.
    //
    // TWO 4-BYTE LOADS, never one float2: q is a permuted view whose row base
    // the op's contract does not align (design 1.4 / 7), and ld.global.v2 on a
    // 4-byte-aligned address is an unspecified launch failure, not a slowdown.
    // The sector count against global memory is identical, so the split costs
    // nothing. Rows past S read row 0 and are zeroed -- out-of-range is a
    // first-class term here, never a consequence of the mask.
    float qa[NKS][4];
    {
        const float * qra = (const float *) (q_base + (oka ? ia : 0)*q_nb1 + h*q_nb2 + b*q_nb3);
        const float * qrc = (const float *) (q_base + (okc ? ic : 0)*q_nb1 + h*q_nb2 + b*q_nb3);
        const int     t   = lane%4;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
            const int d0 = 8*ks + t;         // a_k(0) = a_k(1)
            const int d1 = d0 + 4;           // a_k(2) = a_k(3)
            qa[ks][0] = oka ? fa_tf32::to_tf32(qra[d0]) : 0.0f;   // row a, col t
            qa[ks][1] = okc ? fa_tf32::to_tf32(qrc[d0]) : 0.0f;   // row c, col t
            qa[ks][2] = oka ? fa_tf32::to_tf32(qra[d1]) : 0.0f;   // row a, col t+4
            qa[ks][3] = okc ? fa_tf32::to_tf32(qrc[d1]) : 0.0f;   // row c, col t+4
        }
    }

    // Mask ROW pointers, hoisted out of the tile loop: this lane's two query
    // rows are fixed for the life of the block, so the only per-element index
    // left inside the unrolled bodies is an int32 column. (spec 3.4: modulo
    // broadcast over heads and batch, and mne1 -- not S -- is the row stride.)
    const half * mrow_a = nullptr;
    const half * mrow_c = nullptr;
    if (mask) {
        const int64_t mhb = mne0*mne1*((h % mne2) + mne2*(b % mne3));
        mrow_a = mask + mhb + mne0*(oka ? ia : 0);
        mrow_c = mask + mhb + mne0*(okc ? ic : 0);
    }

    // K/V base for this (kv head, batch); the tile loop advances by BK rows.
    const char * kp0 = k_base + hk*k_nb2 + b*k_nb3;
    const char * vp0 = v_base + hk*v_nb2 + b*v_nb3;

    float oacc[NDB][4];
#pragma unroll
    for (int db = 0; db < NDB; ++db) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            oacc[db][l] = 0.0f;
        }
    }
    float mrun_a = -INFINITY, mrun_c = -INFINITY;   // running row maxima
    float lrun_a = 0.0f,      lrun_c = 0.0f;        // running sums of exp

    for (int64_t j0 = 0; j0 < S_kv; j0 += BK) {
        // Live key columns of THIS tile. `nk` is the out-of-range predicate in
        // its column form and it is a first-class term, not a consequence of
        // the mask: with mask == NULL a pad column would read mv = 0, its p
        // would be NON-zero, and it would inflate the row sum (design 2).
        const int    nk   = (int) min((int64_t) BK, S_kv - j0);
        const half * ma_t = mrow_a ? mrow_a + j0 : nullptr;
        const half * mc_t = mrow_c ? mrow_c + j0 : nullptr;

        // Tile-liveness pre-pass. Only a BOOL survives it: caching the 16 mask
        // values across the staging cost 16 registers and put the kernel into
        // spill, so the score phase below re-fetches them. The mask is 2 bytes
        // an element, warm in L1, and re-read by all 32 query heads; a spilling
        // accumulator is neither.
        bool anylive = false;
        {
            const int cj = 2*(lane%4);
#pragma unroll
            for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = 8*nb + cj + e;
                    if (col < nk) {
                        anylive = anylive ||
                                  (oka && (!ma_t || __half2float(ma_t[col]) != -INFINITY)) ||
                                  (okc && (!mc_t || __half2float(mc_t[col]) != -INFINITY));
                    }
                }
            }
        }

        // ONE barrier does two jobs: it publishes the liveness verdict AND it
        // establishes that every warp has finished reading last tile's K/V, so
        // the staging below is safe. The scan above touches nothing but the
        // mask, which is what makes the merge legal -- three barriers per tile
        // became two.
        if (!__syncthreads_or(anylive)) {
            // Whole tile dead for all 64 rows. With sliding_window = 128 and S
            // in the thousands this is the large majority of tiles. The verdict
            // is block-uniform, so every barrier below stays uniform.
            continue;
        }

        // Stage K and V, swizzled, ALREADY ROUNDED TO TF32: every element is
        // read back once per warp, so converting here is a quarter of the cvt
        // work and bit-identical (cvt.rna is idempotent on a tf32 value). The
        // write is 32 consecutive columns of one row per warp, and an XOR by a
        // mask below 32 permutes within that aligned block -- conflict-free.
        {
            const int    jj0 = tid/D;    // 0 at D = 128
            const int    dd  = tid%D;    // tid at D = 128
            const char * kp  = kp0 + (j0 + jj0)*k_nb1;
            const char * vp  = vp0 + (j0 + jj0)*v_nb1;
#pragma unroll
            for (int it = 0; it < BK/RPI; ++it) {
                const int jj = jj0 + it*RPI;
                float kval = 0.0f;
                float vval = 0.0f;
                if (jj < nk) {
                    kval = fa_tf32::to_tf32(((const float *) kp)[dd]);
                    vval = fa_tf32::to_tf32(((const float *) vp)[dd]);
                }
                const int off = jj*D + (fa_tf32::pcol(dd) ^ fa_tf32::swz(jj));
                Ksh[off] = kval;
                Vsh[off] = vval;
                kp += RPI*k_nb1;
                vp += RPI*v_nb1;
            }
        }

        // (C) K/V tile staged
        __syncthreads();

        // S = Q @ K^T. B operand: n = key = the tile ROW, k = d -> access shape
        // PA (8 rows x 4 consecutive columns), conflict-free under swz.
        float sacc[NNB][4];
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                sacc[nb][l] = 0.0f;
            }
            const int kr = 8*nb + fa_tf32::b_n(0);
#pragma unroll
            for (int ks = 0; ks < NKS; ++ks) {
                // One 8-byte load where a linear layout needs two: the B
                // fragment's {t, t+4} k-pair is adjacent under pcol.
                const float2 kb2 = fa_tf32::pair<D>(Ksh, kr, ks, lane%4);
                float kb[2];
                kb[0] = kb2.x;
                kb[1] = kb2.y;
                fa_tf32::mma_m16n8k8(sacc[nb], qa[ks], kb);
            }
        }

        // Score, then row max. The mask is re-fetched here and the -INF test is
        // on THE MASK VALUE, never on the computed score; the 16-bit `dead` map
        // carries that verdict into the p loop for one register instead of the
        // sixteen a cached mv array cost. The four lanes sharing a query row
        // differ only in lane bits 0 and 1, so a fixed two-step butterfly
        // finishes the reduction and leaves all four holding the same BITS
        // (fmaxf and + are commutative, so each lane computes the same value,
        // not merely an equal one).
        unsigned int dead   = 0;
        float        lmax_a = -INFINITY;
        float        lmax_c = -INFINITY;
        {
            const int cj = 2*(lane%4);
#pragma unroll
            for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int    col = 8*nb + cj + (l%2);
                    const half * mr  = (l < 2) ? ma_t : mc_t;
                    const bool   ok  = (l < 2) ? oka : okc;
                    float mv = -INFINITY;
                    if (ok && col < nk) {
                        mv = mr ? __half2float(mr[col]) : 0.0f;
                    }
                    const float s = fa_tf32::score(sacc[nb][l], scale, mv);
                    sacc[nb][l]   = s;
                    if (mv == -INFINITY) {
                        dead |= 1u << (4*nb + l);
                    }
                    // fmaxf ignores a NaN operand, so a masked element cannot
                    // become the row max however its (unused) dot came out.
                    if (l < 2) {
                        lmax_a = fmaxf(lmax_a, s);
                    } else {
                        lmax_c = fmaxf(lmax_c, s);
                    }
                }
            }
        }
        lmax_a = fmaxf(lmax_a, __shfl_xor_sync(0xffffffff, lmax_a, 1));
        lmax_a = fmaxf(lmax_a, __shfl_xor_sync(0xffffffff, lmax_a, 2));
        lmax_c = fmaxf(lmax_c, __shfl_xor_sync(0xffffffff, lmax_c, 1));
        lmax_c = fmaxf(lmax_c, __shfl_xor_sync(0xffffffff, lmax_c, 2));

        const float mnew_a = fmaxf(mrun_a, lmax_a);
        const float mnew_c = fmaxf(mrun_c, lmax_c);

        // exp(-INF - finite) is 0 in IEEE-754; spelled out rather than
        // evaluated, because -use_fast_math is on for this backend. A row that
        // has still seen nothing keeps corr = 1 (an exact no-op on an all-zero
        // accumulator) and never reaches __expf(-INF - -INF).
        const float corr_a = (mnew_a == -INFINITY) ? 1.0f
                           : ((mrun_a == -INFINITY) ? 0.0f : __expf(mrun_a - mnew_a));
        const float corr_c = (mnew_c == -INFINITY) ? 1.0f
                           : ((mrun_c == -INFINITY) ? 0.0f : __expf(mrun_c - mnew_c));

        // P, in place: the score tile C fragment IS the next mma A fragment
        // (design 1.2), so nothing round-trips through shared. A masked or
        // out-of-range element gets the literal +0.0f whatever TF32 rounding did
        // to its dot. The row sum is over the UNROUNDED p, which is what both
        // the f32 reference and the cuBLAS chain do: they normalise in f32 and
        // round only inside the matmul.
        float psum_a = 0.0f;
        float psum_c = 0.0f;
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float mn = (l < 2) ? mnew_a : mnew_c;
                const float p  = ((dead >> (4*nb + l)) & 1u) ? 0.0f : __expf(sacc[nb][l] - mn);
                sacc[nb][l]    = fa_tf32::to_tf32(p);
                if (l < 2) {
                    psum_a += p;
                } else {
                    psum_c += p;
                }
            }
        }
        psum_a += __shfl_xor_sync(0xffffffff, psum_a, 1);
        psum_a += __shfl_xor_sync(0xffffffff, psum_a, 2);
        psum_c += __shfl_xor_sync(0xffffffff, psum_c, 1);
        psum_c += __shfl_xor_sync(0xffffffff, psum_c, 2);

        lrun_a = lrun_a*corr_a + psum_a;
        lrun_c = lrun_c*corr_c + psum_c;

#pragma unroll
        for (int db = 0; db < NDB; ++db) {
            oacc[db][0] *= corr_a;
            oacc[db][1] *= corr_a;
            oacc[db][2] *= corr_c;
            oacc[db][3] *= corr_c;
        }

        // O += P @ V. B operand: n = d = the tile COLUMN, k = key -> access
        // shape PB (4 rows x 8 consecutive columns), also conflict-free under
        // swz. nb outermost so the d-blocks are INDEPENDENT accumulator chains:
        // latency here is hidden by ILP, not by occupancy.
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
            // The score tile came out of the QK mma as a C fragment and the PV
            // mma wants it as an A fragment, and for tf32 those are different
            // lane maps (fattn-train.cuh). Eight intra-group shuffles, no shared
            // round-trip, no barrier.
            float pa[4];
            fa_tf32::c_to_a(pa, sacc[nb]);

            const int key0 = 8*nb + fa_tf32::b_k(0);
            const int key1 = 8*nb + fa_tf32::b_k(1);
            const int vof0 = key0*D;
            const int vof1 = key1*D;
            const int vz0  = fa_tf32::swz(key0);
            const int vz1  = fa_tf32::swz(key1);
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
                // n = d = the tile COLUMN: two different ROWS, so this one stays
                // two 4-byte loads under any layout.
                const int dcol = fa_tf32::pcol(8*db + fa_tf32::b_n(0));
                float vb[2];
                vb[0] = Vsh[vof0 + (dcol ^ vz0)];
                vb[1] = Vsh[vof1 + (dcol ^ vz1)];
                fa_tf32::mma_m16n8k8(oacc[db], pa, vb);
            }
        }

        mrun_a = mnew_a;
        mrun_c = mnew_c;
    }

    // Packed O | LSE. The C-fragment layout hands a thread two rows x two
    // adjacent columns per d-block, so the store is 8 rows x 32 B where v1's
    // lane = d gave 4 sectors -- deliberate (design 7): it happens once per
    // block, not once per tile, and the tile is what buys the matmul.
    //
    // spec 4.4: a fully-masked query row yields O = 0 and LSE = 0, never NaN.
    // ggml builds the packed tensor gradient as ggml_scale(packed, 0.0f), and
    // 0 * NaN is NaN.
    const int dj = 2*(lane%4);
    if (oka) {
        float * orow = o_data + D*(h + Nh*(ia + S*b));
        if (lrun_a > 0.0f) {
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
                orow[8*db + dj    ] = oacc[db][0]/lrun_a;
                orow[8*db + dj + 1] = oacc[db][1]/lrun_a;
            }
        } else {
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
                orow[8*db + dj    ] = 0.0f;
                orow[8*db + dj + 1] = 0.0f;
            }
        }
        // One value per row, and the C layout gives each row to four lanes:
        // exactly one of them writes it.
        if ((lane & 3) == 0) {
            lse_data[h + Nh*(ia + S*b)] = (lrun_a > 0.0f) ? (mrun_a + __logf(lrun_a)) : 0.0f;
        }
    }
    if (okc) {
        float * orow = o_data + D*(h + Nh*(ic + S*b));
        if (lrun_c > 0.0f) {
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
                orow[8*db + dj    ] = oacc[db][2]/lrun_c;
                orow[8*db + dj + 1] = oacc[db][3]/lrun_c;
            }
        } else {
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
                orow[8*db + dj    ] = 0.0f;
                orow[8*db + dj + 1] = 0.0f;
            }
        }
        if ((lane & 3) == 0) {
            lse_data[h + Nh*(ic + S*b)] = (lrun_c > 0.0f) ? (mrun_c + __logf(lrun_c)) : 0.0f;
        }
    }
#else
    // Pre-Ampere: the instantiation still has to COMPILE for every arch in
    // CMAKE_CUDA_ARCHITECTURES (75;80;86;89;90;120a), so this is a stub. It is
    // never launched -- the host-side dispatch sends pre-sm_80 devices to the v1
    // kernels and traces the reason (design 3.3).
    GGML_UNUSED_VARS(q_base, k_base, v_base, mask, o_data, lse_data,
                     q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3,
                     mne0, mne1, mne2, mne3, S, S_kv, Nh, G, scale);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

bool ggml_cuda_flash_attn_train_supported(const ggml_tensor * op) {
    if (op->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * q    = op->src[0];
    const ggml_tensor * k    = op->src[1];
    const ggml_tensor * v    = op->src[2];
    const ggml_tensor * mask = op->src[3];

    if (!q || !k || !v) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || v->nb[0] != sizeof(float)) {
        return false;
    }

    const int64_t D = q->ne[0];
    if (D != 64 && D != 128) {
        return false;   // v1: the two head dims the trainer and the tests use
    }
    if (k->ne[0] != D || v->ne[0] != D) {
        return false;
    }
    if (k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2]) {
        return false;
    }
    if (k->ne[2] <= 0 || q->ne[2] % k->ne[2] != 0) {
        return false;
    }
    if (q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        return false;
    }
    if (mask) {
        if (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask)) {
            return false;
        }
        if (mask->ne[0] != k->ne[1] || mask->ne[1] < q->ne[1]) {
            return false;
        }
        if (mask->ne[2] <= 0 || mask->ne[3] <= 0) {
            return false;
        }
        if (q->ne[2] % mask->ne[2] != 0 || q->ne[3] % mask->ne[3] != 0) {
            return false;
        }
    }

    // op_params slots 1 and 2 are reserved (max_bias, logit_softcap). Nothing
    // sets them today; if something ever does, fall back rather than ignore it.
    float rsv1 = 0.0f;
    float rsv2 = 0.0f;
    memcpy(&rsv1, (const float *) op->op_params + 1, sizeof(float));
    memcpy(&rsv2, (const float *) op->op_params + 2, sizeof(float));
    if (rsv1 != 0.0f || rsv2 != 0.0f) {
        return false;
    }

    return true;
}

void ggml_cuda_flash_attn_train(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(ggml_cuda_flash_attn_train_supported(dst));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(dst) == ggml_flash_attn_train_nelements(q));

    const int64_t D    = q->ne[0];
    const int64_t S    = q->ne[1];
    const int64_t Nh   = q->ne[2];
    const int64_t Bn   = q->ne[3];
    const int64_t S_kv = k->ne[1];
    const int64_t Nkv  = k->ne[2];
    const int64_t G    = Nh/Nkv;

    // Resolved ONCE (design 3.3): the launch below, the trace line and the
    // parity tool's `prec` column all read this one answer.
    const int                    cc   = ggml_cuda_info().devices[ctx.device].cc;
    const fa_train_prec_resolved prec = fa_train_resolve_prec(dst, cc, /*back =*/ false);
    g_fa_train_last_prec[0] = prec.label;

    static bool traced_fwd = false;
    fa_train_trace(traced_fwd, "flash_attn_train (forward)", D, S, S_kv, Nh, Nkv, Bn,
                   mask != nullptr, prec.label);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t stream = ctx.stream();

    const size_t  offs_lse = ggml_flash_attn_train_lse_offset(q);
    const int64_t n_o      = D*Nh*S*Bn;

    float * o_data   = (float *) dst->data;
    float * lse_data = (float *) ((char *) dst->data + offs_lse);

    // spec 2.2: zero the alignment gap between the O and LSE regions. No
    // kernel writes it, ggml-alloc hands out reused buffers, and the packed
    // tensor's gradient is ggml_scale(packed, 0.0f) -- 0 * inf/NaN is NaN.
    const size_t gap_beg = (size_t) n_o * sizeof(float);
    if (offs_lse > gap_beg) {
        CUDA_CHECK(cudaMemsetAsync((char *) dst->data + gap_beg, 0, offs_lse - gap_beg, stream));
    }

    if (S <= 0) {
        return;
    }

    const dim3 grid((unsigned) ((S + FA_TRAIN_NWARPS - 1)/FA_TRAIN_NWARPS),
                    (unsigned) Nh, (unsigned) Bn);
    const dim3 block(WARP_SIZE, FA_TRAIN_NWARPS);

    const char * q_base = (const char *) q->data;
    const char * k_base = (const char *) k->data;
    const char * v_base = (const char *) v->data;
    const half * m_base = mask ? (const half *) mask->data : nullptr;

    const int64_t mne0 = mask ? mask->ne[0] : 1;
    const int64_t mne1 = mask ? mask->ne[1] : 1;
    const int64_t mne2 = mask ? mask->ne[2] : 1;
    const int64_t mne3 = mask ? mask->ne[3] : 1;

    if (prec.mode == FA_TRAIN_PREC_TF32) {
        // D == 128 is guaranteed by the resolver -- D = 64 is a written-down
        // scope limit, not an oversight, and it falls back to v1 above.
        const dim3 grid_tf32((unsigned) ((S + FA_TRAIN_TF32_BQ - 1)/FA_TRAIN_TF32_BQ),
                             (unsigned) Nh, (unsigned) Bn);
        const dim3 block_tf32(WARP_SIZE, FA_TRAIN_TF32_NW);
        fa_train_fwd_tf32<128><<<grid_tf32, block_tf32, 0, stream>>>(
                q_base, k_base, v_base, m_base, o_data, lse_data,
                (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],
                (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
                (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],
                mne0, mne1, mne2, mne3, S, S_kv, Nh, G, scale);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    switch (D) {
        case 128:
            fa_train_fwd_f32<128><<<grid, block, 0, stream>>>(
                q_base, k_base, v_base, m_base, o_data, lse_data,
                (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],
                (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
                (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],
                mne0, mne1, mne2, mne3, S, S_kv, Nh, G, scale);
            break;
        case 64:
            fa_train_fwd_f32<64><<<grid, block, 0, stream>>>(
                q_base, k_base, v_base, m_base, o_data, lse_data,
                (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],
                (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
                (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],
                mne0, mne1, mne2, mne3, S, S_kv, Nh, G, scale);
            break;
        default:
            GGML_ABORT("ggml_cuda_flash_attn_train: unsupported head dim %d", (int) D);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ─── backward ───────────────────────────────────────────────────────────────
//
// CUDA backward for GGML_OP_FLASH_ATTN_TRAIN_BACK. Behavioural reference is the
// CPU impl in ggml-cpu/ops.cpp (ggml_compute_forward_flash_attn_train_back_f32);
// the maths is spec section 4.2, the kernel split is spec section 8.2.
//
// Three kernels, no floating point atomics anywhere (spec 3.3):
//
//   1. delta   D_i = rowsum(dO * O), one warp per query row, into a [Nh,S,B]
//              pool scratch. Computed once instead of twice, and unlike the
//              CPU path a separate kernel costs no barrier.
//   2. dkdv    one block per (kv tile, kv head, batch). GQA is folded here:
//              the block loops g = 0..G-1 ASCENDING, then query tiles
//              ascending, accumulating dK/dV in registers and writing each
//              output element exactly once. This is where a factor-of-G error
//              would live, and where atomics would destroy determinism.
//   3. dq      one block per (query tile, head, batch), travelling in the
//              FORWARD direction over kv tiles. Splitting dQ into its own
//              same-direction kernel is precisely the atomics-free trick from
//              flash-attention issue #1172.
//
// Neither 2 nor 3 materialises anything S**2: both recompute s and P per tile
// from Q, K, LSE and the mask, and both skip a tile whose mask block is
// entirely -INF (with sliding_window = 128 that is the large majority).
//
// Properties, and how they are held:
//
//   * MASKED POSITIONS CONTRIBUTE EXACTLY 0.0f, bitwise. A -INF mask entry is
//     branched out before exp is ever called, so P is the literal 0.0f and
//     dS = P*(dP - D_i) is the literal 0.0f. A dead key column therefore
//     leaves dK/dV at the +0.0f they were initialised to.
//   * DETERMINISTIC. Every output element is written once by one thread from a
//     private register accumulator; every reduction runs in a fixed order
//     (butterfly shuffle, then tiles ascending, then lanes ascending). Two runs
//     on the same GPU with the same inputs are bitwise identical.
//   * dfwd's LSE region is IGNORED rather than asserted zero (spec 9.7): a
//     future differentiated consumer of the LSE view would make it non-zero and
//     the backward would still be correct.
//   * The two alignment gaps in the packed dQ|dK|dV output are explicitly
//     zeroed (spec 3.2), for one rule rather than two.
//
// Block shape mirrors the forward: FA_TRAIN_BWD_NWARPS warps, one "own" row per
// warp (a kv row in dkdv, a query row in dq), the OTHER side staged
// FA_TRAIN_BWD_TILE rows at a time in shared memory. Inside a tile the lane is
// the far-side index during the dot phase and the head-dim index during the
// accumulate phase, the two connected by a __shfl_sync broadcast -- the same
// two-role trick the forward kernel uses.

#define FA_TRAIN_BWD_NWARPS 4          // "own" rows per block
#define FA_TRAIN_BWD_TILE   WARP_SIZE  // staged far-side rows per tile

// 1. delta[h,s,b] = sum_d dO[d,h,s,b] * O[d,h,s,b].
//    O and dO share the packed forward's [D,Nh,S,B] layout, so row r of both is
//    at D*r with r = h + Nh*(s + S*b) -- the very index LSE uses.
template <int D>
static __global__ void __launch_bounds__(FA_TRAIN_BWD_NWARPS*WARP_SIZE, 1)
fa_train_bwd_delta_f32(
        const float * __restrict__ o_data,
        const float * __restrict__ do_data,
        float       * __restrict__ delta,
        const int64_t nrows) {

    constexpr int NV = D/WARP_SIZE;

    const int     lane = threadIdx.x;
    const int64_t r    = (int64_t) blockIdx.x*FA_TRAIN_BWD_NWARPS + threadIdx.y;

    // r is warp-uniform, so the whole warp leaves together and the reduction
    // below never runs with a partially exited warp.
    if (r >= nrows) {
        return;
    }

    const float * orow = o_data  + (size_t) D*r;
    const float * drow = do_data + (size_t) D*r;

    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < NV; ++c) {
        s = fmaf(orow[lane + WARP_SIZE*c], drow[lane + WARP_SIZE*c], s);
    }
    s = warp_reduce_sum<WARP_SIZE>(s);

    if (lane == 0) {
        delta[r] = s;
    }
}

// 2. dK / dV. One block per (kv tile, kv head, batch); one kv row per warp.
template <int D>
static __global__ void __launch_bounds__(FA_TRAIN_BWD_NWARPS*WARP_SIZE, 1)
fa_train_bwd_dkdv_f32(
        const char  * __restrict__ q_base,
        const char  * __restrict__ k_base,
        const char  * __restrict__ v_base,
        const half  * __restrict__ mask,
        const float * __restrict__ do_data,
        const float * __restrict__ lse_data,
        const float * __restrict__ delta,
        float       * __restrict__ dk_data,
        float       * __restrict__ dv_data,
        const int64_t q_nb1, const int64_t q_nb2, const int64_t q_nb3,
        const int64_t k_nb1, const int64_t k_nb2, const int64_t k_nb3,
        const int64_t v_nb1, const int64_t v_nb2, const int64_t v_nb3,
        const int64_t mne0, const int64_t mne1, const int64_t mne2, const int64_t mne3,
        const int64_t S, const int64_t S_kv, const int64_t Nh, const int64_t Nkv,
        const int64_t G, const float scale) {

    constexpr int NV = D/WARP_SIZE;      // accumulator floats per lane
    constexpr int LD = D + 1;            // odd stride: bank-conflict free
    constexpr int BQ = FA_TRAIN_BWD_TILE;
    constexpr int NW = FA_TRAIN_BWD_NWARPS;
    constexpr int NT = NW*WARP_SIZE;

    __shared__ float Qsh   [BQ*LD];
    __shared__ float dOsh  [BQ*LD];
    __shared__ float Ksh   [NW*LD];
    __shared__ float Vsh   [NW*LD];
    __shared__ float lse_sh[BQ];
    __shared__ float del_sh[BQ];
    __shared__ int   sh_live[NW];

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = w*WARP_SIZE + lane;

    const int64_t j0b = (int64_t) blockIdx.x*NW;     // first kv row of the block
    const int64_t j   = j0b + w;                     // this warp's kv row
    const int64_t hk  = blockIdx.y;
    const int64_t b   = blockIdx.z;

    const bool jok = j < S_kv;

    // The block's NW K/V rows are staged once and stay resident. Published by
    // the first __syncthreads() inside the loop.
    for (int idx = tid; idx < NW*D; idx += NT) {
        const int     jj   = idx/D;
        const int     dd   = idx%D;
        const int64_t jrow = j0b + jj;
        float kval = 0.0f;
        float vval = 0.0f;
        if (jrow < S_kv) {
            const float * krow = (const float *) (k_base + jrow*k_nb1 + hk*k_nb2 + b*k_nb3);
            const float * vrow = (const float *) (v_base + jrow*v_nb1 + hk*v_nb2 + b*v_nb3);
            kval = krow[dd];
            vval = vrow[dd];
        }
        Ksh[jj*LD + dd] = kval;
        Vsh[jj*LD + dd] = vval;
    }

    float dk_acc[NV];
    float dv_acc[NV];
#pragma unroll
    for (int c = 0; c < NV; ++c) {
        dk_acc[c] = 0.0f;
        dv_acc[c] = 0.0f;
    }

    // GQA (spec 3.3): query heads ASCENDING, then query tiles ascending. The
    // order is the contract, not an implementation detail -- it is what makes
    // an A/B against --attn exact interpretable.
    for (int64_t g = 0; g < G; ++g) {
        const int64_t h = hk*G + g;

        for (int64_t i0 = 0; i0 < S; i0 += BQ) {
            // (A) the previous tile's reads of Qsh/dOsh are done, and on the
            //     very first pass this publishes the K/V staging above
            __syncthreads();

            const int64_t i   = i0 + lane;
            const bool    iok = jok && (i < S);

            // spec 3.4: modulo broadcast, and mne1 (not S) is the row stride
            float mv = 0.0f;
            if (iok && mask) {
                const int64_t midx = j + mne0*(i + mne1*((h % mne2) + mne2*(b % mne3)));
                mv = __half2float(mask[midx]);
            }
            const bool live = iok && !(mv == -INFINITY);

            const int anyw = __any_sync(0xffffffff, live);
            if (lane == 0) {
                sh_live[w] = anyw;
            }

            // (B) tile liveness known to the whole block
            __syncthreads();

            int block_live = 0;
#pragma unroll
            for (int t = 0; t < NW; ++t) {
                block_live |= sh_live[t];
            }
            if (!block_live) {
                // block_live is block-uniform, so every thread takes this and
                // the __syncthreads() at (A) stays uniform.
                continue;
            }

            const int64_t nq = min((int64_t) BQ, S - i0);
            for (int idx = tid; idx < BQ*D; idx += NT) {
                const int ii = idx/D;
                const int dd = idx%D;
                float qval = 0.0f;
                float oval = 0.0f;
                if (ii < nq) {
                    const float * qrow = (const float *) (q_base + (i0 + ii)*q_nb1 + h*q_nb2 + b*q_nb3);
                    qval = qrow[dd];
                    oval = do_data[(size_t) D*(h + Nh*((i0 + ii) + S*b)) + dd];
                }
                Qsh [ii*LD + dd] = qval;
                dOsh[ii*LD + dd] = oval;
            }
            for (int idx = tid; idx < BQ; idx += NT) {
                float lv = 0.0f;
                float dv = 0.0f;
                if (idx < nq) {
                    const int64_t r = h + Nh*((i0 + idx) + S*b);
                    lv = lse_data[r];
                    dv = delta[r];
                }
                lse_sh[idx] = lv;
                del_sh[idx] = dv;
            }

            // (C) query tile staged
            __syncthreads();

            float p  = 0.0f;
            float cf = 0.0f;
            if (live) {
                const float * qs = Qsh + lane*LD;
                const float * ks = Ksh + w*LD;
                float dot = 0.0f;
#pragma unroll 8
                for (int d = 0; d < D; ++d) {
                    dot = fmaf(qs[d], ks[d], dot);
                }
                // the mask is additive AFTER the scale, which is why dQ/dK
                // carry a scale factor and dV does not
                const float s = scale*dot + mv;
                p = __expf(s - lse_sh[lane]);
                if (p != 0.0f) {
                    const float * os = dOsh + lane*LD;
                    const float * vs = Vsh  + w*LD;
                    float dp = 0.0f;
#pragma unroll 8
                    for (int d = 0; d < D; ++d) {
                        dp = fmaf(os[d], vs[d], dp);
                    }
                    // dP is finite by construction and the multiply by P kills
                    // it where P is 0. Never reordered into (dP - D_i) times
                    // something that can be inf.
                    cf = scale*(p*(dp - del_sh[lane]));
                }
            }

            // Fixed lane order, so the accumulation order is fixed too. pj is
            // warp-uniform, so the skip below never splits the warp.
#pragma unroll 4
            for (int jj = 0; jj < WARP_SIZE; ++jj) {
                const float pj = __shfl_sync(0xffffffff, p, jj, WARP_SIZE);
                if (pj == 0.0f) {
                    continue;   // masked, or vanished: contributes exactly nothing
                }
                const float cj = __shfl_sync(0xffffffff, cf, jj, WARP_SIZE);
                const float * os = dOsh + jj*LD;
                const float * qs = Qsh  + jj*LD;
#pragma unroll
                for (int c = 0; c < NV; ++c) {
                    dv_acc[c] = fmaf(pj, os[lane + WARP_SIZE*c], dv_acc[c]);
                    dk_acc[c] = fmaf(cj, qs[lane + WARP_SIZE*c], dk_acc[c]);
                }
            }
        }
    }

    if (!jok) {
        return;
    }

    // written exactly once, by exactly this thread
    const size_t off = (size_t) D*(j + S_kv*(hk + Nkv*b));
#pragma unroll
    for (int c = 0; c < NV; ++c) {
        dk_data[off + lane + WARP_SIZE*c] = dk_acc[c];
        dv_data[off + lane + WARP_SIZE*c] = dv_acc[c];
    }
}

// 3. dQ. One block per (query tile, head, batch); one query row per warp;
//    kv tiles walked in the FORWARD direction (issue #1172).
template <int D>
static __global__ void __launch_bounds__(FA_TRAIN_BWD_NWARPS*WARP_SIZE, 1)
fa_train_bwd_dq_f32(
        const char  * __restrict__ q_base,
        const char  * __restrict__ k_base,
        const char  * __restrict__ v_base,
        const half  * __restrict__ mask,
        const float * __restrict__ do_data,
        const float * __restrict__ lse_data,
        const float * __restrict__ delta,
        float       * __restrict__ dq_data,
        const int64_t q_nb1, const int64_t q_nb2, const int64_t q_nb3,
        const int64_t k_nb1, const int64_t k_nb2, const int64_t k_nb3,
        const int64_t v_nb1, const int64_t v_nb2, const int64_t v_nb3,
        const int64_t mne0, const int64_t mne1, const int64_t mne2, const int64_t mne3,
        const int64_t S, const int64_t S_kv, const int64_t Nh, const int64_t G,
        const float scale) {

    constexpr int NV = D/WARP_SIZE;
    constexpr int LD = D + 1;
    constexpr int BK = FA_TRAIN_BWD_TILE;
    constexpr int NW = FA_TRAIN_BWD_NWARPS;
    constexpr int NT = NW*WARP_SIZE;

    __shared__ float Ksh [BK*LD];
    __shared__ float Vsh [BK*LD];
    __shared__ float Qsh [NW*LD];
    __shared__ float dOsh[NW*LD];
    __shared__ int   sh_live[NW];

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = w*WARP_SIZE + lane;

    const int64_t i  = (int64_t) blockIdx.x*NW + w;   // query row
    const int64_t h  = blockIdx.y;
    const int64_t b  = blockIdx.z;
    const int64_t hk = h/G;                           // GQA kv head

    const bool active = i < S;

    float lse_i = 0.0f;
    float del_i = 0.0f;
    if (active) {
        const float * qrow = (const float *) (q_base + i*q_nb1 + h*q_nb2 + b*q_nb3);
        const float * orow = do_data + (size_t) D*(h + Nh*(i + S*b));
#pragma unroll
        for (int c = 0; c < NV; ++c) {
            Qsh [w*LD + lane + WARP_SIZE*c] = qrow[lane + WARP_SIZE*c];
            dOsh[w*LD + lane + WARP_SIZE*c] = orow[lane + WARP_SIZE*c];
        }
        const int64_t r = h + Nh*(i + S*b);
        lse_i = lse_data[r];
        del_i = delta[r];
    }
    // Qsh/dOsh rows are read back only by the warp that wrote them.
#ifndef GGML_USE_HIP
    __syncwarp();   // a HIP wavefront runs in lockstep, so the barrier is implicit there
#endif // GGML_USE_HIP

    float dq_acc[NV];
#pragma unroll
    for (int c = 0; c < NV; ++c) {
        dq_acc[c] = 0.0f;
    }

    for (int64_t j0 = 0; j0 < S_kv; j0 += BK) {
        // (A) previous tile's compute is done: safe to overwrite K/V and flags
        __syncthreads();

        const int64_t j   = j0 + lane;
        const bool    jok = active && (j < S_kv);

        float mv = 0.0f;
        if (jok && mask) {
            const int64_t midx = j + mne0*(i + mne1*((h % mne2) + mne2*(b % mne3)));
            mv = __half2float(mask[midx]);
        }
        const bool live = jok && !(mv == -INFINITY);

        const int anyw = __any_sync(0xffffffff, live);
        if (lane == 0) {
            sh_live[w] = anyw;
        }

        // (B) tile liveness known to the whole block
        __syncthreads();

        int block_live = 0;
#pragma unroll
        for (int t = 0; t < NW; ++t) {
            block_live |= sh_live[t];
        }
        if (!block_live) {
            continue;
        }

        const int64_t nk = min((int64_t) BK, S_kv - j0);
        for (int idx = tid; idx < BK*D; idx += NT) {
            const int jj = idx/D;
            const int dd = idx%D;
            float kval = 0.0f;
            float vval = 0.0f;
            if (jj < nk) {
                const float * krow = (const float *) (k_base + (j0 + jj)*k_nb1 + hk*k_nb2 + b*k_nb3);
                const float * vrow = (const float *) (v_base + (j0 + jj)*v_nb1 + hk*v_nb2 + b*v_nb3);
                kval = krow[dd];
                vval = vrow[dd];
            }
            Ksh[jj*LD + dd] = kval;
            Vsh[jj*LD + dd] = vval;
        }

        // (C) K/V tile staged
        __syncthreads();

        float cf = 0.0f;
        if (live) {
            const float * qs = Qsh + w*LD;
            const float * ks = Ksh + lane*LD;
            float dot = 0.0f;
#pragma unroll 8
            for (int d = 0; d < D; ++d) {
                dot = fmaf(qs[d], ks[d], dot);
            }
            const float s = scale*dot + mv;
            const float p = __expf(s - lse_i);
            if (p != 0.0f) {
                const float * os = dOsh + w*LD;
                const float * vs = Vsh  + lane*LD;
                float dp = 0.0f;
#pragma unroll 8
                for (int d = 0; d < D; ++d) {
                    dp = fmaf(os[d], vs[d], dp);
                }
                cf = scale*(p*(dp - del_i));
            }
        }

#pragma unroll 4
        for (int jj = 0; jj < WARP_SIZE; ++jj) {
            const float cj = __shfl_sync(0xffffffff, cf, jj, WARP_SIZE);
            if (cj == 0.0f) {
                continue;   // dS is exactly 0 here; adding 0*K changes nothing
            }
            const float * ks = Ksh + jj*LD;
#pragma unroll
            for (int c = 0; c < NV; ++c) {
                dq_acc[c] = fmaf(cj, ks[lane + WARP_SIZE*c], dq_acc[c]);
            }
        }
    }

    if (!active) {
        return;
    }

    float * out = dq_data + (size_t) D*(i + S*(h + Nh*b));
#pragma unroll
    for (int c = 0; c < NV; ++c) {
        out[lane + WARP_SIZE*c] = dq_acc[c];
    }
}

// ─── TF32 backward (roadmap R1) ─────────────────────────────────────────────
//
// docs/plans/fattn-train-tf32-design.md 1.3. Identical contract to the three v1
// kernels above -- the same atomics-free dQ / dK+dV split, the same -INF rule,
// the same fixed schedules, the same "written once by one thread" store -- with
// the SEVEN matmuls of the backward (the split recomputes S and dP in both
// kernels, so it is seven, not five) moved onto
// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32.
//
// `delta` is deliberately NOT rewritten: D_i = rowsum(dO . O) is 2 FLOP per
// element and purely memory-bound, so casting it as a matmul would waste 16x the
// work to extract a diagonal (design 1.3). It stays scalar f32 and stays
// bit-exact between the two modes.
//
// Everything that is not a matmul stays f32: the out-of-range predicate, the
// mask fetch, the -INF branch to a literal +0.0f, exp, and
// dS = scale*(P*(dP - D_i)) with the *P outermost. The two helpers below are
// what keep dQ's and dK/dV's recompute the SAME function -- design 7 names two
// divergent implementations of one recompute as the miserable-to-bisect failure
// ("dQ agrees, dK does not"), and the loop bodies genuinely cannot be shared
// because the schedules differ.

static __device__ __forceinline__ float fa_bwd_p(
        const float acc, const float scale, const float mv, const float lse) {
    if (mv == -INFINITY) {
        // masked OR out of range -- the two are bitwise the same thing here, and
        // the test is on THE MASK VALUE, never on the computed score, so the
        // literal +0.0f survives whatever TF32 rounding did to the dot.
        return 0.0f;
    }
    return __expf(fa_tf32::score(acc, scale, mv) - lse);
}

static __device__ __forceinline__ float fa_bwd_ds(
        const float p, const float dp, const float del, const float scale) {
    // Never reassociated: the * p stays outermost so a zero P kills whatever
    // (dP - D_i) came out as. A dead key column therefore leaves dK bitwise
    // +0.0, exactly as in v1.
    return (p == 0.0f) ? 0.0f : scale*(p*(dp - del));
}

// dK / dV. One block per (16-key tile, kv head, GQA split, batch); NW warps
// arranged as DH d-halves x QS query-splits of the SAME staged query tile.
//
// THE SHAPE OF THIS KERNEL IS SET BY OCCUPANCY AND PER-TILE OVERHEAD, NOT BY
// MATMUL COUNT. Three measurements on a 5090 at the S = 625 bench rung, which is
// the campaign's gate, and each one moved the design:
//
//   1. The first version -- 2 warps, one block per (key tile, kv head), the GQA
//      loop inside -- ran 0.716 ms, i.e. 12.0 TFLOP/s, while the dQ kernel beside
//      it hit 39.5 on near-identical work per unit of score-matrix area. dQ
//      launches 640 blocks x 4 warps; this launched 320 x 2, so it had 3.8 warps
//      per SM against dQ's 12 and nothing to hide shared-load and mma latency
//      with. FIX: DH x QS warps, and a GQA split across blocks -> 0.407 ms.
//   2. ptxas was using 248 registers under __launch_bounds__(..., 1), which caps
//      the SM at 2 blocks however small the tile is. Asking for 3 and partially
//      unrolling the two k-slice loops lands on 160 registers with ZERO spill
//      -> 0.361 ms. (Asking for 3 WITHOUT the partial unroll spills 128 bytes and
//      gives most of it back; the dQ kernel wants the opposite and is left at 1.)
//   3. Halving the matmul count did NOT help. A role split -- one warp owning
//      dV and computing S^T, the other owning dK and computing dP^T, P^T handed
//      over through shared -- does 128 mma per 16-key x 16-query tile against
//      this shape's 192, and measured 0.765 ms against 0.671 for the whole site.
//      It buys arithmetic and pays a barrier and a worse staging-to-compute
//      ratio, and this kernel is bound by the second thing. So the redundant
//      S^T / dP^T that the d-split costs is deliberately KEPT, and what got
//      optimised instead is the per-tile overhead: two barriers rather than
//      three (the liveness vote IS the barrier that protects the staging
//      buffers), and float2 fragment loads under pcol.
//
// K and V are the A operands and live in SHARED, not registers: holding them as
// fragments would be another 128 registers per thread on top of the
// accumulators. Under pcol their A-fragment read is two 8-byte loads at rows
// {g, g+8} -- eight rows by four consecutive columns, access shape PA and
// conflict-free under the swizzle. Note this is NOT the design's PC shape: that
// was derived from the C/D lane map, and for tf32 the A map is a different
// function (see fattn-train.cuh).

#define FA_TRAIN_TF32_DKDV_DH 2                                        // d-splits
#define FA_TRAIN_TF32_DKDV_QS 2                                        // query-splits
#define FA_TRAIN_TF32_DKDV_NW (FA_TRAIN_TF32_DKDV_DH*FA_TRAIN_TF32_DKDV_QS)
#define FA_TRAIN_TF32_DKDV_BJ 16                                       // kv rows owned per block
#define FA_TRAIN_TF32_DKDV_BI (8*FA_TRAIN_TF32_DKDV_QS)                // query rows per staged tile

// Target block count for the GQA split. A pure function of the launch shape --
// no SM count, no device query -- so the same graph splits the same way on every
// GPU and the run-twice bitwise gate keeps its meaning. 1024 is 3 blocks on each
// of ~340 SMs, i.e. comfortably past the shared-memory occupancy limit of the
// largest part this is likely to meet, and the clamp to a DIVISOR of G is what
// keeps the per-block head count exact. Measured alternatives at the three bench
// rungs: 512 is worse at S = 625 and 1250, 2048 is better at 1250 and worse at
// 3000 (its partial buffer costs more than the extra blocks buy once the grid is
// already 1504 wide).
static int fa_train_dkdv_gsplit(const int64_t nkv_tiles, const int64_t Nkv,
                                const int64_t Bn, const int64_t G) {
    const int64_t base = nkv_tiles*Nkv*Bn;
    if (base <= 0) {
        return 1;
    }
    const int64_t want = (1024 + base - 1)/base;
    int gs = 1;
    for (int64_t cand = 1; cand <= G; ++cand) {
        if (G % cand == 0 && cand <= want) {
            gs = (int) cand;
        }
    }
    return gs;
}

template <int D>
static __global__ void __launch_bounds__(FA_TRAIN_TF32_DKDV_NW*WARP_SIZE, 3)
fa_train_bwd_dkdv_tf32(
        const char  * __restrict__ q_base,
        const char  * __restrict__ k_base,
        const char  * __restrict__ v_base,
        const half  * __restrict__ mask,
        const float * __restrict__ do_data,
        const float * __restrict__ lse_data,
        const float * __restrict__ delta,
        float       * __restrict__ dk_data,
        float       * __restrict__ dv_data,
        const int64_t q_nb1, const int64_t q_nb2, const int64_t q_nb3,
        const int64_t k_nb1, const int64_t k_nb2, const int64_t k_nb3,
        const int64_t v_nb1, const int64_t v_nb2, const int64_t v_nb3,
        const int64_t mne0, const int64_t mne1, const int64_t mne2, const int64_t mne3,
        const int64_t S, const int64_t S_kv, const int64_t Nh, const int64_t Nkv,
        const int64_t G, const float scale,
        const int GS, const int64_t part_stride) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int DH  = FA_TRAIN_TF32_DKDV_DH;
    constexpr int QS  = FA_TRAIN_TF32_DKDV_QS;
    constexpr int NW  = FA_TRAIN_TF32_DKDV_NW;
    constexpr int BJ  = FA_TRAIN_TF32_DKDV_BJ;
    constexpr int BI  = FA_TRAIN_TF32_DKDV_BI;
    constexpr int NT  = NW*WARP_SIZE;
    constexpr int DW  = D/DH;    // output d columns owned per warp
    constexpr int NDB = DW/8;    // 8-wide d-blocks of the dK/dV accumulators
    constexpr int NKS = D/8;     // k-slices of the S^T and dP^T reductions
    constexpr int NGR = D/8;     // 8-column staging groups per row

    static_assert(BJ == 16,          "one mma M tile of keys per block");
    static_assert(D % DH == 0,       "the d split must divide the head dim");
    static_assert(DW % 8 == 0,       "the d split must land on 8-wide blocks");
    static_assert(BI == 8*QS,        "one mma n-block of queries per query split");
    static_assert(BI*NGR % NT == 0,  "the staging loop assumes an exact divide");
    static_assert(BJ*NGR % NT == 0,  "the K/V staging assumes an exact divide");
    static_assert(BJ*DW*DH <= BI*D,  "the reduction scratch must fit the staging buffers");

    __shared__ __align__(16) float Ksh [BJ*D];
    __shared__ __align__(16) float Vsh [BJ*D];
    __shared__ __align__(16) float Qsh [BI*D];
    __shared__ __align__(16) float dOsh[BI*D];
    __shared__ float lse_sh[BI];
    __shared__ float del_sh[BI];

    const int lane  = threadIdx.x;
    const int w     = threadIdx.y;
    const int tid   = w*WARP_SIZE + lane;
    const int g4    = lane/4;      // groupID of every fragment map
    const int t4    = lane%4;      // thread-in-group of every fragment map
    const int dh    = w % DH;      // this warp's d half
    const int qs    = w / DH;      // this warp's query split
    const int dbase = dh*DW;       // this warp's first output d column
    const int qbase = qs*8;        // this warp's first query column of the tile

    const int64_t j0b  = (int64_t) blockIdx.x*BJ;
    const int64_t hk   = blockIdx.y / GS;
    const int64_t gsp  = blockIdx.y % GS;
    const int64_t gper = G/GS;                 // query heads handled by this block
    const int64_t gbeg = gsp*gper;
    const int64_t b    = blockIdx.z;

    // The C/D map gives this lane two KEY rows 8 apart (M = keys here, because
    // the score tile is produced transposed -- design 1.2's orientation rule).
    const int64_t ja  = j0b + g4;
    const int64_t jc  = j0b + 8 + g4;
    const bool    oka = ja < S_kv;
    const bool    okc = jc < S_kv;

    // mne0 is k->ne[1] == S_kv; mstride*BI cannot approach 2^31 at any geometry
    // this op is reachable at, and an int32 column index is what keeps the
    // unrolled bodies out of 64-bit address arithmetic.
    const int mstride = (int) mne0;

    // The block's BJ K/V rows: staged once, resident for the whole sweep,
    // ROUNDED TO TF32 at the staging. Every element is read back once per warp
    // per k-slice per query tile, so converting here is a fraction of the cvt
    // work and bit-identical (cvt.rna is idempotent on a tf32 value) -- and it
    // is the same rounding the forward applied to the same tensors, which is
    // what makes the recomputed S^T the forward's S.
    //
    // A whole 8-column group per thread per step, in and out as float2: pcol
    // sends source columns {c, c+4} to adjacent destinations {2c', 2c'+1}, so
    // eight elements are four 8-byte global loads and four 8-byte shared stores
    // where the scalar form needs sixteen. The 8-byte global load is what
    // fa_train_view_8aligned guarantees in the dispatch.
#pragma unroll
    for (int it = 0; it < BJ*NGR/NT; ++it) {
        const int idx = tid + it*NT;
        const int jj  = idx/NGR;
        const int gr  = (idx%NGR)*8;
        float2 ks[4];
        float2 vs[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            ks[e] = make_float2(0.0f, 0.0f);
            vs[e] = make_float2(0.0f, 0.0f);
        }
        if (j0b + jj < S_kv) {
            const float2 * kr = (const float2 *) ((const char *) k_base + hk*k_nb2 + b*k_nb3 + (j0b + jj)*k_nb1);
            const float2 * vr = (const float2 *) ((const char *) v_base + hk*v_nb2 + b*v_nb3 + (j0b + jj)*v_nb1);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                ks[e] = kr[gr/2 + e];
                vs[e] = vr[gr/2 + e];
            }
        }
        // pcol sends source column c to 2*(c&3) | ((c>>2)&1), so destination
        // pair p holds source columns {p, p+4} -- one 8-byte store each.
        const float * ksf = (const float *) ks;
        const float * vsf = (const float *) vs;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int off = jj*D + ((gr + 2*p) ^ fa_tf32::swz(jj));
            *(float2 *) (Ksh + off) = make_float2(fa_tf32::to_tf32(ksf[p]),
                                                  fa_tf32::to_tf32(ksf[p + 4]));
            *(float2 *) (Vsh + off) = make_float2(fa_tf32::to_tf32(vsf[p]),
                                                  fa_tf32::to_tf32(vsf[p + 4]));
        }
    }

    float dkacc[NDB][4];
    float dvacc[NDB][4];
#pragma unroll
    for (int db = 0; db < NDB; ++db) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            dkacc[db][l] = 0.0f;
            dvacc[db][l] = 0.0f;
        }
    }

    // GQA (spec 3.3): query heads ASCENDING, then query tiles ascending. The
    // order is the contract, not an implementation detail. With GS > 1 the head
    // range is this block's slice of it and the gsum pass restores the whole,
    // also ascending.
    for (int64_t g = gbeg; g < gbeg + gper; ++g) {
        const int64_t h  = hk*G + g;
        const char  * qh = q_base + h*q_nb2 + b*q_nb3;
        const float * dh_row = do_data + (size_t) D*h;   // + D*Nh*(i + S*b)
        const int64_t rh = h + Nh*S*b;                   // + Nh*i

        // Mask COLUMN pointers for this lane's two keys: the key is fixed for
        // the life of the block, so the only per-element index left inside the
        // unrolled bodies is an int32 query offset scaled by mstride.
        const half * mca = nullptr;
        const half * mcc = nullptr;
        const half * mhb = nullptr;
        if (mask) {
            mhb = mask + mne0*mne1*((h % mne2) + mne2*(b % mne3));
            mca = mhb + (oka ? ja : 0);
            mcc = mhb + (okc ? jc : 0);
        }

        for (int64_t i0 = 0; i0 < S; i0 += BI) {
            const int    ni   = (int) min((int64_t) BI, S - i0);
            const half * ma_t = mca ? mca + mne0*i0 : nullptr;
            const half * mc_t = mcc ? mcc + mne0*i0 : nullptr;

            // Tile liveness. Scanned COALESCED over the whole BJ x BI block --
            // consecutive threads take consecutive keys, which is 16 contiguous
            // halves per mask row -- rather than each lane checking the elements
            // it happens to own, which would be a stride-mne0 gather. With
            // mask == NULL nothing is dead and the scan is skipped entirely.
            //
            // It runs BEFORE the barrier and touches nothing but the mask, which
            // is what lets one __syncthreads_or do the work of two barriers plus
            // a shared vote: it publishes the verdict AND establishes that every
            // warp has finished reading last tile's Qsh/dOsh. Three barriers per
            // tile became two, on a kernel whose measured limit is per-tile
            // overhead rather than matmul.
            bool anylive = (mask == nullptr);
            if (mask) {
                const half * mtile = mhb + j0b + mne0*i0;
#pragma unroll
                for (int it = 0; it < (BJ*BI + NT - 1)/NT; ++it) {
                    const int idx = tid + it*NT;
                    if (idx < BJ*BI) {
                        const int jj = idx%BJ;
                        const int ii = idx/BJ;
                        if (ii < ni && j0b + jj < S_kv) {
                            anylive = anylive || (__half2float(mtile[jj + mstride*ii]) != -INFINITY);
                        }
                    }
                }
            }
            if (!__syncthreads_or(anylive)) {
                // block-uniform, so every barrier below stays uniform
                continue;
            }

            // Stage the query tile: Q and dO both swizzled, both TF32-rounded
            // (both are B operands), plus the row-wise LSE and D_i. Same whole-
            // group float2 form as the K/V staging above.
#pragma unroll
            for (int it = 0; it < BI*NGR/NT; ++it) {
                const int idx = tid + it*NT;
                const int ii  = idx/NGR;
                const int gr  = (idx%NGR)*8;
                float2 qs2[4];
                float2 os2[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    qs2[e] = make_float2(0.0f, 0.0f);
                    os2[e] = make_float2(0.0f, 0.0f);
                }
                if (ii < ni) {
                    const float2 * qr = (const float2 *) (qh + (i0 + ii)*q_nb1);
                    const float2 * or_ = (const float2 *) (dh_row + (size_t) D*Nh*((i0 + ii) + S*b));
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        qs2[e] = qr [gr/2 + e];
                        os2[e] = or_[gr/2 + e];
                    }
                }
                const float * qf = (const float *) qs2;
                const float * of = (const float *) os2;
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int off = ii*D + ((gr + 2*p) ^ fa_tf32::swz(ii));
                    *(float2 *) (Qsh  + off) = make_float2(fa_tf32::to_tf32(qf[p]),
                                                           fa_tf32::to_tf32(qf[p + 4]));
                    *(float2 *) (dOsh + off) = make_float2(fa_tf32::to_tf32(of[p]),
                                                           fa_tf32::to_tf32(of[p + 4]));
                }
            }
            for (int idx = tid; idx < BI; idx += NT) {
                float lv = 0.0f;
                float dv = 0.0f;
                if (idx < ni) {
                    lv = lse_data[rh + Nh*(i0 + idx)];
                    dv = delta   [rh + Nh*(i0 + idx)];
                }
                lse_sh[idx] = lv;
                del_sh[idx] = dv;
            }

            // query tile staged
            __syncthreads();

            // S^T = K @ Q^T. A = K from shared as float2 pairs under pcol;
            // B = Q with n = query = the tile ROW, also a pair. Both are access
            // shape PA and conflict-free. This warp owns query columns
            // [qbase, qbase+8) and nothing else.
            const int qr = qbase + g4;
            float sacc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 8
            for (int ks = 0; ks < NKS; ++ks) {
                const float2 aa = fa_tf32::pair<D>(Ksh, g4,     ks, t4);
                const float2 ac = fa_tf32::pair<D>(Ksh, g4 + 8, ks, t4);
                const float2 bb = fa_tf32::pair<D>(Qsh, qr,     ks, t4);
                float ka[4];
                float qb[2];
                ka[0] = aa.x;   // row g,   k = t
                ka[1] = ac.x;   // row g+8, k = t
                ka[2] = aa.y;   // row g,   k = t+4
                ka[3] = ac.y;   // row g+8, k = t+4
                qb[0] = bb.x;
                qb[1] = bb.y;
                fa_tf32::mma_m16n8k8(sacc, ka, qb);
            }

            // P^T = exp(s - LSE_i), in place. The -INF test is on the mask
            // value, and an out-of-range query column is bitwise a masked one.
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int    ql = qbase + 2*t4 + (l%2);
                const bool   ok = (l < 2) ? oka : okc;
                const half * mr = (l < 2) ? ma_t : mc_t;
                float mv = -INFINITY;
                if (ok && ql < ni) {
                    mv = mr ? __half2float(mr[mstride*ql]) : 0.0f;
                }
                sacc[l] = fa_bwd_p(sacc[l], scale, mv, lse_sh[ql]);
            }

            // dP^T = V @ dO^T. Same operand shapes as S^T with V for K and dO
            // for Q; the reduction is over the FULL dv, which is why both
            // d-split warps compute it.
            float dpacc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 8
            for (int ks = 0; ks < NKS; ++ks) {
                const float2 aa = fa_tf32::pair<D>(Vsh,  g4,     ks, t4);
                const float2 ac = fa_tf32::pair<D>(Vsh,  g4 + 8, ks, t4);
                const float2 bb = fa_tf32::pair<D>(dOsh, qr,     ks, t4);
                float va[4];
                float ob[2];
                va[0] = aa.x;
                va[1] = ac.x;
                va[2] = aa.y;
                va[3] = ac.y;
                ob[0] = bb.x;
                ob[1] = bb.y;
                fa_tf32::mma_m16n8k8(dpacc, va, ob);
            }

            // dS^T = scale * P^T * (dP^T - D_i), then both score tiles become
            // A operands (C map -> A map is eight intra-group shuffles for tf32,
            // not the identity it is for ggml's f16 tiles).
            float pa[4];
            float da[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int   ql = qbase + 2*t4 + (l%2);
                const float p  = sacc[l];
                dpacc[l] = fa_tf32::to_tf32(fa_bwd_ds(p, dpacc[l], del_sh[ql], scale));
                sacc [l] = fa_tf32::to_tf32(p);
            }
            fa_tf32::c_to_a(pa, sacc);
            fa_tf32::c_to_a(da, dpacc);

            // dV += P^T @ dO and dK += dS^T @ Q. B operand n = d = the tile
            // COLUMN -> access shape PB: the two elements are on different ROWS,
            // so this one stays two 4-byte loads under any layout.
            {
                const int r0 = qbase + t4;       // b_k(0)
                const int r1 = r0 + 4;           // b_k(1)
                const int z0 = fa_tf32::swz(r0);
                const int z1 = fa_tf32::swz(r1);
#pragma unroll
                for (int db = 0; db < NDB; ++db) {
                    const int dcol = fa_tf32::pcol(dbase + 8*db + g4);
                    float ob[2];
                    float qb[2];
                    ob[0] = dOsh[r0*D + (dcol ^ z0)];
                    ob[1] = dOsh[r1*D + (dcol ^ z1)];
                    qb[0] = Qsh [r0*D + (dcol ^ z0)];
                    qb[1] = Qsh [r1*D + (dcol ^ z1)];
                    fa_tf32::mma_m16n8k8(dvacc[db], pa, ob);
                    fa_tf32::mma_m16n8k8(dkacc[db], da, qb);
                }
            }

            // The compute above is the last read of Qsh/dOsh for this tile; the
            // next iteration's __syncthreads_or is what protects them.
        }
    }

    // The QS query splits accumulated into the same (key, d) outputs, so they
    // are summed here -- split 0's registers plus split 1's, ALWAYS in that
    // order, through the now-dead Q and dO staging buffers, one thread per
    // element. Fixed-order shared reduction, never an atomicAdd and never a
    // __reduce_add: design 4's determinism contract is what makes an A/B against
    // --attn exact interpretable at all.
    __syncthreads();
#pragma unroll
    for (int src = 1; src < QS; ++src) {
        if (qs == src) {
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int slot = dh*(BJ*DW) + (8*(l/2) + g4)*DW + 8*db + 2*t4 + (l%2);
                    Qsh [slot] = dkacc[db][l];
                    dOsh[slot] = dvacc[db][l];
                }
            }
        }
        __syncthreads();
        if (qs == 0) {
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int slot = dh*(BJ*DW) + (8*(l/2) + g4)*DW + 8*db + 2*t4 + (l%2);
                    dkacc[db][l] += Qsh [slot];
                    dvacc[db][l] += dOsh[slot];
                }
            }
        }
        __syncthreads();
    }
    if (qs != 0) {
        return;
    }

    // Written exactly once, by exactly this thread. The C/D map gives the lane
    // two key rows and two adjacent d columns per 8-wide block. With GS > 1 this
    // is the split's own partial plane and fa_train_bwd_gsum_f32 folds them.
    float * dk_out = dk_data + gsp*part_stride;
    float * dv_out = dv_data + gsp*part_stride;
    const int dj = 2*t4;
    if (oka) {
        const size_t off = (size_t) D*(ja + S_kv*(hk + Nkv*b));
#pragma unroll
        for (int db = 0; db < NDB; ++db) {
            const size_t o = off + dbase + 8*db + dj;
            dk_out[o    ] = dkacc[db][0];
            dk_out[o + 1] = dkacc[db][1];
            dv_out[o    ] = dvacc[db][0];
            dv_out[o + 1] = dvacc[db][1];
        }
    }
    if (okc) {
        const size_t off = (size_t) D*(jc + S_kv*(hk + Nkv*b));
#pragma unroll
        for (int db = 0; db < NDB; ++db) {
            const size_t o = off + dbase + 8*db + dj;
            dk_out[o    ] = dkacc[db][2];
            dk_out[o + 1] = dkacc[db][3];
            dv_out[o    ] = dvacc[db][2];
            dv_out[o + 1] = dvacc[db][3];
        }
    }
#else
    GGML_UNUSED_VARS(q_base, k_base, v_base, mask, do_data, lse_data, delta,
                     dk_data, dv_data,
                     q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3,
                     mne0, mne1, mne2, mne3, S, S_kv, Nh, Nkv, G, scale,
                     GS, part_stride);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// Folds the GQA split's GS partial planes into dK / dV, ascending split order,
// one thread per output element. No atomics; the order is the contract.
static __global__ void fa_train_bwd_gsum_f32(
        const float * __restrict__ part,
        float       * __restrict__ out,
        const int64_t n, const int64_t stride, const int GS) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float s = part[i];
    for (int g = 1; g < GS; ++g) {
        s += part[i + (int64_t) g*stride];
    }
    out[i] = s;
}

// dQ. One block per (BQ-query tile, head, batch), kv tiles walked in the FORWARD
// direction -- the same-direction split from flash-attention issue #1172 is
// unchanged, and it is what makes an atomics-free dQ possible at all.
//
// 4 warps arranged as 2 query groups x 2 d-halves (design 1.4): the accumulator
// alone is 64 registers at D = 128, and dQ additionally needs P and dP live at
// the same instant for dS = scale*P*(dP - D_i). Splitting the accumulator's
// d-range across two warps halves it to 32; the two warps of a query group
// recompute S and dP redundantly (both reduce over the full D) and write
// disjoint output columns, so there is no cross-warp reduction.
//
// Q lives in REGISTERS as A-fragments, loaded exactly the way the forward loads
// them -- same rows, same to_tf32, same k-slice order -- so this kernel's
// recomputed S is the forward's S rather than merely close to it. dO lives in
// SHARED because a second 64-register fragment set on top of Q would spill.
//
// TUNED THE OPPOSITE WAY TO dK/dV, and the measurements are why. At S = 625 this
// kernel already runs its 7.55 GFLOP in 0.184 ms -- about 41 TFLOP/s, 2.5x the
// mma issue floor and better than the cuBLAS chain it replaces -- so it is
// ILP-bound where dK/dV was occupancy-bound. Both of dK/dV's wins therefore
// LOSE here and are deliberately not applied: __launch_bounds__(..., 3) with a
// partial unroll took it to 0.223 ms, and pcol'd float2 fragment loads took it
// to 0.223 as well. It keeps full unrolls, a minimum-blocks hint of 1, and the
// plain swizzle. The cost, recorded rather than rounded down because design 7
// calls any spill a design failure: 255 registers with 144 BYTES of spill on
// sm_120a. Every config that removed the spill measured slower end to end.

#define FA_TRAIN_TF32_DQ_QG 2                       // query groups per block
#define FA_TRAIN_TF32_DQ_DH 2                       // d-splits per query group
#define FA_TRAIN_TF32_DQ_NW (FA_TRAIN_TF32_DQ_QG*FA_TRAIN_TF32_DQ_DH)
#define FA_TRAIN_TF32_DQ_BQ (FA_TRAIN_TF32_DQ_QG*16)
#define FA_TRAIN_TF32_DQ_BK 16                      // keys staged per tile

template <int D>
static __global__ void __launch_bounds__(FA_TRAIN_TF32_DQ_NW*WARP_SIZE, 1)
fa_train_bwd_dq_tf32(
        const char  * __restrict__ q_base,
        const char  * __restrict__ k_base,
        const char  * __restrict__ v_base,
        const half  * __restrict__ mask,
        const float * __restrict__ do_data,
        const float * __restrict__ lse_data,
        const float * __restrict__ delta,
        float       * __restrict__ dq_data,
        const int64_t q_nb1, const int64_t q_nb2, const int64_t q_nb3,
        const int64_t k_nb1, const int64_t k_nb2, const int64_t k_nb3,
        const int64_t v_nb1, const int64_t v_nb2, const int64_t v_nb3,
        const int64_t mne0, const int64_t mne1, const int64_t mne2, const int64_t mne3,
        const int64_t S, const int64_t S_kv, const int64_t Nh, const int64_t G,
        const float scale) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int QG  = FA_TRAIN_TF32_DQ_QG;
    constexpr int DHV = FA_TRAIN_TF32_DQ_DH;
    constexpr int NW  = FA_TRAIN_TF32_DQ_NW;
    constexpr int BQ  = FA_TRAIN_TF32_DQ_BQ;
    constexpr int BK  = FA_TRAIN_TF32_DQ_BK;
    constexpr int NT  = NW*WARP_SIZE;
    constexpr int DW  = D/DHV;   // output d columns owned per warp
    constexpr int NDB = DW/8;    // 8-wide d-blocks of the dQ accumulator
    constexpr int NKS = D/8;     // k-slices of the S and dP reductions
    constexpr int NNB = BK/8;    // 8-wide key blocks of the score tile

    static_assert(D % DHV == 0,   "the d split must divide the head dim");
    static_assert(DW % 8 == 0,    "the d split must land on 8-wide blocks");
    static_assert(BK % 8 == 0,    "the score tile is a whole number of n-blocks");
    static_assert(BQ*D % NT == 0, "the dO staging assumes an exact divide");
    static_assert(BK*D % NT == 0, "the K/V staging assumes an exact divide");

    // 8-byte aligned because the A / PA fragment reads are float2 under pcol.
    __shared__ __align__(16) float dOsh[BQ*D];
    __shared__ __align__(16) float Ksh [BK*D];
    __shared__ __align__(16) float Vsh [BK*D];

    const int lane  = threadIdx.x;
    const int w     = threadIdx.y;
    const int tid   = w*WARP_SIZE + lane;
    const int g4    = lane/4;
    const int t4    = lane%4;
    const int qg    = w/DHV;         // this warp's query group
    const int dbase = (w%DHV)*DW;    // this warp's first output d column

    const int64_t qb0 = (int64_t) blockIdx.x*BQ;   // block's first query row
    const int64_t q0  = qb0 + 16*qg;               // warp's first query row
    const int64_t h   = blockIdx.y;
    const int64_t b   = blockIdx.z;
    const int64_t hk  = h/G;

    const int64_t ia  = q0 + g4;
    const int64_t ic  = q0 + 8 + g4;
    const bool    oka = ia < S;
    const bool    okc = ic < S;

    const int mstride = (int) mne0;

    // Q -> A-fragment registers, byte for byte the forward's load.
    float qa[NKS][4];
    {
        const float * qra = (const float *) (q_base + (oka ? ia : 0)*q_nb1 + h*q_nb2 + b*q_nb3);
        const float * qrc = (const float *) (q_base + (okc ? ic : 0)*q_nb1 + h*q_nb2 + b*q_nb3);
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
            const int d0 = 8*ks + t4;
            const int d1 = d0 + 4;
            qa[ks][0] = oka ? fa_tf32::to_tf32(qra[d0]) : 0.0f;
            qa[ks][1] = okc ? fa_tf32::to_tf32(qrc[d0]) : 0.0f;
            qa[ks][2] = oka ? fa_tf32::to_tf32(qra[d1]) : 0.0f;
            qa[ks][3] = okc ? fa_tf32::to_tf32(qrc[d1]) : 0.0f;
        }
    }

    // dO for the WHOLE block -> shared, swizzled, TF32-rounded. Staged once:
    // it is an A operand of dP, re-read per key tile, and holding it in
    // registers on top of Q would spill.
    {
        const int ii0 = tid/D;
        const int dd  = tid%D;
        constexpr int RPI = NT/D;
#pragma unroll
        for (int it = 0; it < BQ/RPI; ++it) {
            const int     ii = ii0 + it*RPI;
            const int64_t ir = qb0 + ii;
            float oval = 0.0f;
            if (ir < S) {
                oval = fa_tf32::to_tf32(do_data[(size_t) D*(h + Nh*(ir + S*b)) + dd]);
            }
            dOsh[ii*D + (dd ^ fa_tf32::swz(ii))] = oval;
        }
    }

    float lse_a = 0.0f, lse_c = 0.0f;
    float del_a = 0.0f, del_c = 0.0f;
    if (oka) {
        const int64_t r = h + Nh*(ia + S*b);
        lse_a = lse_data[r];
        del_a = delta   [r];
    }
    if (okc) {
        const int64_t r = h + Nh*(ic + S*b);
        lse_c = lse_data[r];
        del_c = delta   [r];
    }

    // Mask ROW pointers for this lane's two query rows (row stride mne0).
    const half * mrow_a = nullptr;
    const half * mrow_c = nullptr;
    const half * mtile0 = nullptr;
    if (mask) {
        const half * mhb = mask + mne0*mne1*((h % mne2) + mne2*(b % mne3));
        mrow_a = mhb + mne0*(oka ? ia : 0);
        mrow_c = mhb + mne0*(okc ? ic : 0);
        mtile0 = mhb + mne0*qb0;      // block's first query row, column 0
    }

    const char * kp0 = k_base + hk*k_nb2 + b*k_nb3;
    const char * vp0 = v_base + hk*v_nb2 + b*v_nb3;

    float dqacc[NDB][4];
#pragma unroll
    for (int db = 0; db < NDB; ++db) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            dqacc[db][l] = 0.0f;
        }
    }

    for (int64_t j0 = 0; j0 < S_kv; j0 += BK) {
        const int    nk   = (int) min((int64_t) BK, S_kv - j0);
        const half * ma_t = mrow_a ? mrow_a + j0 : nullptr;
        const half * mc_t = mrow_c ? mrow_c + j0 : nullptr;

        // Coalesced BQ x BK liveness scan (consecutive threads take consecutive
        // key columns, 16 contiguous halves per mask row).
        bool anylive = (mask == nullptr);
        if (mask) {
            const half * mtile = mtile0 + j0;
#pragma unroll
            for (int it = 0; it < BQ*BK/NT; ++it) {
                const int idx = tid + it*NT;
                const int jj  = idx%BK;
                const int ii  = idx/BK;
                if (jj < nk && qb0 + ii < S) {
                    anylive = anylive || (__half2float(mtile[jj + mstride*ii]) != -INFINITY);
                }
            }
        }

        // ONE barrier for the liveness verdict AND for "every warp is done
        // reading last tile's K/V"; on the first pass it also publishes the dO
        // staging above. The scan touches nothing but the mask, which is what
        // makes the merge legal.
        if (!__syncthreads_or(anylive)) {
            continue;
        }

        // Stage K and V, swizzled, TF32-rounded -- the forward's staging.
        {
            const int jj0 = tid/D;
            const int dd  = tid%D;
            constexpr int RPI = NT/D;
#pragma unroll
            for (int it = 0; it < BK/RPI; ++it) {
                const int jj = jj0 + it*RPI;
                float kval = 0.0f;
                float vval = 0.0f;
                if (jj < nk) {
                    kval = fa_tf32::to_tf32(((const float *) (kp0 + (j0 + jj)*k_nb1))[dd]);
                    vval = fa_tf32::to_tf32(((const float *) (vp0 + (j0 + jj)*v_nb1))[dd]);
                }
                const int off = jj*D + (dd ^ fa_tf32::swz(jj));
                Ksh[off] = kval;
                Vsh[off] = vval;
            }
        }

        // (C) K/V tile staged
        __syncthreads();

        // S = Q @ K^T -- bitwise the forward's product: same A registers, same
        // B tile, same ascending k-slice order.
        float sacc[NNB][4];
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                sacc[nb][l] = 0.0f;
            }
            // NOT pcol'd, deliberately: the same float2 rewrite that pays in the
            // dK/dV kernel MEASURED 0.184 -> 0.223 ms here at S = 625 (it put
            // ptxas back into an 8-byte spill at 255 registers). This kernel is
            // already at ~41 TFLOP/s, i.e. 2.5x the mma issue floor and better
            // than the cuBLAS chain it replaces; it is ILP-bound, not
            // shared-load-bound, and the extra address arithmetic costs more
            // than the halved load count buys.
            const int kr = 8*nb + g4;
            const int kz = fa_tf32::swz(kr);
#pragma unroll
            for (int ks = 0; ks < NKS; ++ks) {
                float kb[2];
                kb[0] = Ksh[kr*D + ((8*ks + t4    ) ^ kz)];
                kb[1] = Ksh[kr*D + ((8*ks + t4 + 4) ^ kz)];
                fa_tf32::mma_m16n8k8(sacc[nb], qa[ks], kb);
            }
        }

        // P = exp(s - LSE_i), in place.
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int    col = 8*nb + 2*t4 + (l%2);
                const bool   ok  = (l < 2) ? oka : okc;
                const half * mr  = (l < 2) ? ma_t : mc_t;
                float mv = -INFINITY;
                if (ok && col < nk) {
                    mv = mr ? __half2float(mr[col]) : 0.0f;
                }
                sacc[nb][l] = fa_bwd_p(sacc[nb][l], scale, mv, (l < 2) ? lse_a : lse_c);
            }
        }

        // dP = dO @ V^T. A = dO from shared (four PA-shaped loads per k-slice,
        // hoisted out of the n-block loop); B = V with n = key = the tile ROW.
        float dpacc[NNB][4];
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                dpacc[nb][l] = 0.0f;
            }
        }
        {
            const int ora = 16*qg + g4;
            const int orc = ora + 8;
            const int oza = fa_tf32::swz(ora);
            const int ozc = fa_tf32::swz(orc);
#pragma unroll
            for (int ks = 0; ks < NKS; ++ks) {
                const int c0 = 8*ks + t4;
                const int c1 = c0 + 4;
                float da[4];
                da[0] = dOsh[ora*D + (c0 ^ oza)];
                da[1] = dOsh[orc*D + (c0 ^ ozc)];
                da[2] = dOsh[ora*D + (c1 ^ oza)];
                da[3] = dOsh[orc*D + (c1 ^ ozc)];
#pragma unroll
                for (int nb = 0; nb < NNB; ++nb) {
                    const int vr = 8*nb + g4;
                    const int vz = fa_tf32::swz(vr);
                    float vb[2];
                    vb[0] = Vsh[vr*D + (c0 ^ vz)];
                    vb[1] = Vsh[vr*D + (c1 ^ vz)];
                    fa_tf32::mma_m16n8k8(dpacc[nb], da, vb);
                }
            }
        }

        // dS = scale * P * (dP - D_i), then dQ += dS @ K with n = d = the tile
        // COLUMN (access shape PB).
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                dpacc[nb][l] = fa_tf32::to_tf32(
                        fa_bwd_ds(sacc[nb][l], dpacc[nb][l], (l < 2) ? del_a : del_c, scale));
            }
            float da[4];
            fa_tf32::c_to_a(da, dpacc[nb]);

            const int r0 = 8*nb + t4;
            const int r1 = r0 + 4;
            const int z0 = fa_tf32::swz(r0);
            const int z1 = fa_tf32::swz(r1);
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
                // n = d = the tile COLUMN: two different ROWS, so this one stays
                // two 4-byte loads under any layout.
                const int dcol = dbase + 8*db + g4;
                float kb[2];
                kb[0] = Ksh[r0*D + (dcol ^ z0)];
                kb[1] = Ksh[r1*D + (dcol ^ z1)];
                fa_tf32::mma_m16n8k8(dqacc[db], da, kb);
            }
        }
    }

    const int dj = 2*t4;
    if (oka) {
        float * out = dq_data + (size_t) D*(ia + S*(h + Nh*b));
#pragma unroll
        for (int db = 0; db < NDB; ++db) {
            out[dbase + 8*db + dj    ] = dqacc[db][0];
            out[dbase + 8*db + dj + 1] = dqacc[db][1];
        }
    }
    if (okc) {
        float * out = dq_data + (size_t) D*(ic + S*(h + Nh*b));
#pragma unroll
        for (int db = 0; db < NDB; ++db) {
            out[dbase + 8*db + dj    ] = dqacc[db][2];
            out[dbase + 8*db + dj + 1] = dqacc[db][3];
        }
    }
#else
    GGML_UNUSED_VARS(q_base, k_base, v_base, mask, do_data, lse_data, delta, dq_data,
                     q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3,
                     mne0, mne1, mne2, mne3, S, S_kv, Nh, G, scale);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

bool ggml_cuda_flash_attn_train_back_supported(const ggml_tensor * op) {
    // src[0..3] and op_params are exactly the forward's, so the shape/type
    // capability check is shared rather than duplicated.
    if (!ggml_cuda_flash_attn_train_supported(op)) {
        return false;
    }

    const ggml_tensor * fwd  = op->src[4];
    const ggml_tensor * dfwd = op->src[5];
    if (!fwd || !dfwd) {
        return false;
    }
    if (fwd->type != GGML_TYPE_F32 || dfwd->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(fwd) || !ggml_is_contiguous(dfwd)) {
        return false;
    }
    if (ggml_nelements(fwd) != ggml_nelements(dfwd)) {
        return false;
    }
    if (ggml_nelements(fwd) != ggml_flash_attn_train_nelements(op->src[0])) {
        return false;
    }

    return true;
}

void ggml_cuda_flash_attn_train_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * fwd  = dst->src[4];
    const ggml_tensor * dfwd = dst->src[5];

    GGML_ASSERT(ggml_cuda_flash_attn_train_back_supported(dst));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // spec 5.2: ggml_scale(packed, 0.0f) must never run in place, and this op
    // must never alias the forward it reads O and LSE out of.
    GGML_ASSERT(dst->data != fwd->data);

    const int64_t D    = q->ne[0];
    const int64_t S    = q->ne[1];
    const int64_t Nh   = q->ne[2];
    const int64_t Bn   = q->ne[3];
    const int64_t S_kv = k->ne[1];
    const int64_t Nkv  = k->ne[2];
    const int64_t G    = Nh/Nkv;

    // Same resolver, same rule, `back = true`. Recorded and traced rather than
    // assumed: the label carries the reason, so a run's arithmetic is readable
    // off the trace instead of inferred from the flag.
    const int                    cc   = ggml_cuda_info().devices[ctx.device].cc;
    const fa_train_prec_resolved prec = fa_train_resolve_prec(dst, cc, /*back =*/ true);
    g_fa_train_last_prec[1] = prec.label;

    static bool traced_bwd = false;
    fa_train_trace(traced_bwd, "flash_attn_train_back (backward)", D, S, S_kv, Nh, Nkv, Bn,
                   mask != nullptr, prec.label);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t stream = ctx.stream();

    size_t  offs_dq = 0;
    size_t  offs_dk = 0;
    size_t  offs_dv = 0;
    int64_t nelem   = 0;
    ggml_flash_attn_train_back_offsets(q, k, v, &offs_dq, &offs_dk, &offs_dv, &nelem);
    GGML_ASSERT(ggml_nelements(dst) == nelem);

    float * dq_data = (float *) ((char *) dst->data + offs_dq);
    float * dk_data = (float *) ((char *) dst->data + offs_dk);
    float * dv_data = (float *) ((char *) dst->data + offs_dv);

    const float * o_data   = (const float *) fwd->data;
    const float * lse_data = (const float *) ((const char *) fwd->data + ggml_flash_attn_train_lse_offset(q));
    const float * do_data  = (const float *) dfwd->data;   // dO in the O region; the
                                                           // LSE region is ignored (spec 9.7)

    // spec 3.2: zero the two alignment gaps, same rule as the forward. Lower
    // stakes here (this output is read only through its three views and is
    // never scaled by zero) -- one rule beats two.
    const int64_t n_q = D*S*Nh*Bn;
    const int64_t n_k = D*S_kv*Nkv*Bn;
    {
        const size_t g0 = (size_t) n_q*sizeof(float);
        if (offs_dk > g0) {
            CUDA_CHECK(cudaMemsetAsync((char *) dst->data + g0, 0, offs_dk - g0, stream));
        }
        const size_t g1 = offs_dk + (size_t) n_k*sizeof(float);
        if (offs_dv > g1) {
            CUDA_CHECK(cudaMemsetAsync((char *) dst->data + g1, 0, offs_dv - g1, stream));
        }
    }

    if (S <= 0 || S_kv <= 0) {
        // Degenerate, and outside the trainer's reach; the defined answer is an
        // all-zero gradient rather than an unwritten buffer.
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
        return;
    }

    const char * q_base = (const char *) q->data;
    const char * k_base = (const char *) k->data;
    const char * v_base = (const char *) v->data;
    const half * m_base = mask ? (const half *) mask->data : nullptr;

    const int64_t mne0 = mask ? mask->ne[0] : 1;
    const int64_t mne1 = mask ? mask->ne[1] : 1;
    const int64_t mne2 = mask ? mask->ne[2] : 1;
    const int64_t mne3 = mask ? mask->ne[3] : 1;

    const int64_t nrows = Nh*S*Bn;
    ggml_cuda_pool_alloc<float> delta(ctx.pool(), (size_t) nrows);

    const dim3 block(WARP_SIZE, FA_TRAIN_BWD_NWARPS);
    const dim3 grid_delta((unsigned) ((nrows + FA_TRAIN_BWD_NWARPS - 1)/FA_TRAIN_BWD_NWARPS), 1, 1);
    const dim3 grid_dkdv ((unsigned) ((S_kv  + FA_TRAIN_BWD_NWARPS - 1)/FA_TRAIN_BWD_NWARPS),
                          (unsigned) Nkv, (unsigned) Bn);
    const dim3 grid_dq   ((unsigned) ((S     + FA_TRAIN_BWD_NWARPS - 1)/FA_TRAIN_BWD_NWARPS),
                          (unsigned) Nh,  (unsigned) Bn);

#define FA_TRAIN_BWD_LAUNCH(DD)                                                              \
    do {                                                                                     \
        fa_train_bwd_delta_f32<DD><<<grid_delta, block, 0, stream>>>(                         \
                o_data, do_data, delta.get(), nrows);                                         \
        fa_train_bwd_dkdv_f32<DD><<<grid_dkdv, block, 0, stream>>>(                           \
                q_base, k_base, v_base, m_base, do_data, lse_data, delta.get(),               \
                dk_data, dv_data,                                                             \
                (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],                   \
                (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],                   \
                (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],                   \
                mne0, mne1, mne2, mne3, S, S_kv, Nh, Nkv, G, scale);                          \
        fa_train_bwd_dq_f32<DD><<<grid_dq, block, 0, stream>>>(                               \
                q_base, k_base, v_base, m_base, do_data, lse_data, delta.get(),               \
                dq_data,                                                                      \
                (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],                   \
                (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],                   \
                (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],                   \
                mne0, mne1, mne2, mne3, S, S_kv, Nh, G, scale);                               \
    } while (0)

    if (prec.mode == FA_TRAIN_PREC_TF32) {
        // D == 128 is guaranteed by the resolver (D = 64 is a written-down scope
        // limit and falls back to v1 above). `delta` is deliberately shared with
        // the f32 path: it is memory-bound, bit-exact, and casting it as a matmul
        // would waste 16x the work to extract a diagonal (design 1.3).
        const dim3 block_dkdv(WARP_SIZE, FA_TRAIN_TF32_DKDV_NW);
        const dim3 block_dq  (WARP_SIZE, FA_TRAIN_TF32_DQ_NW);
        const dim3 grid_dq_tf32  ((unsigned) ((S    + FA_TRAIN_TF32_DQ_BQ   - 1)/FA_TRAIN_TF32_DQ_BQ),
                                  (unsigned) Nh,  (unsigned) Bn);

        // GQA split (see the dK/dV kernel's header): dK/dV's grid is Nkv-wide
        // where dQ's is Nh-wide, and at short S that is the difference between
        // filling the machine and not. GS is a function of the LAUNCH SHAPE
        // alone, never of the device, so a result reproduces elsewhere; it
        // collapses to 1 -- no partial buffer, no second pass -- as soon as the
        // grid is large enough on its own.
        const int64_t nkv_tiles = (S_kv + FA_TRAIN_TF32_DKDV_BJ - 1)/FA_TRAIN_TF32_DKDV_BJ;
        const int     GS        = fa_train_dkdv_gsplit(nkv_tiles, Nkv, Bn, G);
        const int64_t n_kv_el   = D*S_kv*Nkv*Bn;
        const dim3    grid_dkdv_tf32((unsigned) nkv_tiles, (unsigned) (Nkv*GS), (unsigned) Bn);

        ggml_cuda_pool_alloc<float> dkv_part(ctx.pool());
        float * dk_dst = dk_data;
        float * dv_dst = dv_data;
        int64_t part_stride = 0;
        if (GS > 1) {
            dkv_part.alloc((size_t) 2*GS*n_kv_el);
            dk_dst      = dkv_part.get();
            dv_dst      = dkv_part.get() + (size_t) GS*n_kv_el;
            part_stride = n_kv_el;
        }

        fa_train_bwd_delta_f32<128><<<grid_delta, block, 0, stream>>>(
                o_data, do_data, delta.get(), nrows);
        fa_train_bwd_dkdv_tf32<128><<<grid_dkdv_tf32, block_dkdv, 0, stream>>>(
                q_base, k_base, v_base, m_base, do_data, lse_data, delta.get(),
                dk_dst, dv_dst,
                (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],
                (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
                (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],
                mne0, mne1, mne2, mne3, S, S_kv, Nh, Nkv, G, scale, GS, part_stride);
        if (GS > 1) {
            const int64_t nthr = 256;
            const dim3    grid_sum((unsigned) ((n_kv_el + nthr - 1)/nthr), 1, 1);
            fa_train_bwd_gsum_f32<<<grid_sum, (unsigned) nthr, 0, stream>>>(
                    dk_dst, dk_data, n_kv_el, part_stride, GS);
            fa_train_bwd_gsum_f32<<<grid_sum, (unsigned) nthr, 0, stream>>>(
                    dv_dst, dv_data, n_kv_el, part_stride, GS);
        }
        fa_train_bwd_dq_tf32<128><<<grid_dq_tf32, block_dq, 0, stream>>>(
                q_base, k_base, v_base, m_base, do_data, lse_data, delta.get(),
                dq_data,
                (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],
                (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
                (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],
                mne0, mne1, mne2, mne3, S, S_kv, Nh, G, scale);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    switch (D) {
        case 128:
            FA_TRAIN_BWD_LAUNCH(128);
            break;
        case 64:
            FA_TRAIN_BWD_LAUNCH(64);
            break;
        default:
            GGML_ABORT("ggml_cuda_flash_attn_train_back: unsupported head dim %d", (int) D);
    }
#undef FA_TRAIN_BWD_LAUNCH

    CUDA_CHECK(cudaGetLastError());
}
