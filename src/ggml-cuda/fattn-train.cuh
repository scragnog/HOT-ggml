// HOT-Step patch: flash-attn-train
//
// CUDA kernels for GGML_OP_FLASH_ATTN_TRAIN{,_BACK}. Self-contained on
// purpose: no existing fattn-*.cu / fattn-common.cuh file is touched, so the
// inference attention path keeps its exact object code and its recompile blast
// radius. See docs/plans/fattn-train-spec.md sections 2, 4 and 8.
#pragma once

#include "common.cuh"

// Forward: packed O | LSE, tiled online softmax, f32 in/out, f32 accumulate.
void ggml_cuda_flash_attn_train(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Capability check for ggml_backend_cuda_device_supports_op. Shape/type only --
// it is called on graph nodes that have no data yet.
bool ggml_cuda_flash_attn_train_supported(const ggml_tensor * op);

// HOT-Step patch: flash-attn-train
// Backward: three atomics-free kernels (delta / dK+dV / dQ), packed dQ|dK|dV.
void ggml_cuda_flash_attn_train_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Capability check for ggml_backend_cuda_device_supports_op. Shape/type only.
bool ggml_cuda_flash_attn_train_back_supported(const ggml_tensor * op);

// HOT-Step patch: flash-attn-train
// Which arithmetic the last CUDA launch of each direction actually used.
// dir 0 = forward, dir 1 = backward. Returns a static string: "tf32", or
// "f32 (<reason>)" when the request was overridden, or "n/a" before any launch.
// Exposed through the backend registry's get_proc_address as
// "ggml_backend_cuda_fattn_train_last_prec" so the parity tool can print the
// kernel that RAN rather than the flag it asked for (tf32 design 3.3 / 5.3).
const char * ggml_cuda_fattn_train_last_prec(int dir);

// HOT-Step patch: flash-attn-train
//
// TF32 tensor-core primitives for the fused training attention (roadmap R1,
// docs/plans/fattn-train-tf32-design.md section 1.1).
//
// SELF-CONTAINED ON PURPOSE, and the reason turned out to be sharper than the
// design's: mma.cuh carries this exact instruction (line 1089), but its
// `tile<16, 8, float>` lane map describes the **C/D accumulator only**. ggml
// exercises that tile as `tile_C` in mmf / mmq / fattn-mma and nothing else, so
// the map is right where it is used -- but the same tile is also declared as the
// tf32 mma's A operand at mma.cuh:1083, and NOTHING IN GGML CALLS THAT OVERLOAD.
// Its A map has therefore never run. It is wrong, and so was the "A and C are
// the same function" identity the design leaned on (1.2).
//
// The maps below were MEASURED, not read: a standalone probe computed a 16x8x8
// product with every candidate A x B map against a scalar loop and kept the two
// that matched bitwise (max abs err 0.0; the other fourteen were off by ~2). The
// survivor is the PTX ISA's own m16n8k8 .tf32 layout:
//
//   A   16x8 tf32, 4 regs   a0 = (row g,   col t  )   a1 = (row g+8, col t  )
//                           a2 = (row g,   col t+4)   a3 = (row g+8, col t+4)
//   B   8x8  tf32, 2 regs   b0 = (row t,   col g  )   b1 = (row t+4, col g  )
//                           (B is K x N; "row" is the k index, "col" the n index)
//   C/D 16x8 f32,  4 regs   c0,c1 = (row g,   col 2t+{0,1})
//                           c2,c3 = (row g+8, col 2t+{0,1})
//
//   with g = threadIdx.x/4 (groupID) and t = threadIdx.x%4 (thread in group).
//
// Note what that costs: for tf32 the A and C/D maps are DIFFERENT, so a score
// tile computed as C is NOT already an A operand. c_to_a below pays for it in
// eight intra-group shuffles rather than a shared round-trip.
namespace fa_tf32 {
    // C / D accumulator.
    static __device__ __forceinline__ int c_i(const int l) { return 8*(l/2) + (int) (threadIdx.x/4); }
    static __device__ __forceinline__ int c_j(const int l) { return 2*(int) (threadIdx.x%4) + (l%2); }
    // A operand (M x K).
    static __device__ __forceinline__ int a_i(const int l) { return 8*(l%2) + (int) (threadIdx.x/4); }
    static __device__ __forceinline__ int a_k(const int l) { return 4*(l/2) + (int) (threadIdx.x%4); }
    // B operand (K x N).
    static __device__ __forceinline__ int b_n(const int l) { GGML_UNUSED(l); return (int) (threadIdx.x/4); }
    static __device__ __forceinline__ int b_k(const int l) { return 4*l + (int) (threadIdx.x%4); }

    // cvt.rna.tf32.f32 is NOT optional. PTX reads .tf32 operands out of f32
    // registers with the low 13 mantissa bits TRUNCATED, not rounded; one
    // instruction per register roughly halves the input rounding error, which
    // is what cuBLAS does internally and therefore what the 5e-3 parity bar was
    // measured against. Applied to A and B fragments only -- accumulators stay
    // f32, which is the ".f32" in the instruction's type string.
    static __device__ __forceinline__ float to_tf32(const float x) {
#ifdef AMPERE_MMA_AVAILABLE
        unsigned int r;
        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
        return __uint_as_float(r);
#else
        return x;
#endif // AMPERE_MMA_AVAILABLE
    }

    // D[16x8] += A[16x8] @ B[8x8]. TF32 multiply, f32 accumulate.
    static __device__ __forceinline__ void mma_m16n8k8(
            float (&D)[4], const float (&A)[4], const float (&B)[2]) {
#ifdef AMPERE_MMA_AVAILABLE
        int       * Di = (int       *) D;
        const int * Ai = (const int *) A;
        const int * Bi = (const int *) B;
        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Di[0]), "+r"(Di[1]), "+r"(Di[2]), "+r"(Di[3])
            : "r"(Ai[0]), "r"(Ai[1]), "r"(Ai[2]), "r"(Ai[3]), "r"(Bi[0]), "r"(Bi[1]));
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
    }

    // C fragment -> A fragment, for feeding a score tile straight into the next
    // matmul. Both maps hold rows {g, g+8}, so nothing crosses a 4-lane group and
    // the row never moves; only the columns are redistributed, C's {2t, 2t+1} to
    // A's {t, t+4}. Column c lives on lane (c/2) as element (c%2), so each half
    // needs both of that lane's registers and a local select -- a single shuffle
    // cannot do it, because the two receivers 2s and 2s+1 want DIFFERENT
    // registers of the same source s.
    //
    // Deterministic by construction: fixed source lanes, no ballot, full mask,
    // every lane participating.
    static __device__ __forceinline__ void c_to_a(float (&A)[4], const float (&C)[4]) {
#ifdef AMPERE_MMA_AVAILABLE
        const int  t    = (int) (threadIdx.x & 3);
        const int  base = (int) (threadIdx.x & ~3u);
        const int  sLo  = base + (t >> 1);        // holds columns {t,   t+1-ish}: 2*(t>>1) + {0,1}
        const int  sHi  = sLo + 2;                // holds columns 2*((t>>1)+2) + {0,1} = t+4 pair
        const bool hi   = (t & 1) != 0;
        const float l0 = __shfl_sync(0xffffffff, C[0], sLo);
        const float l1 = __shfl_sync(0xffffffff, C[1], sLo);
        const float l2 = __shfl_sync(0xffffffff, C[2], sLo);
        const float l3 = __shfl_sync(0xffffffff, C[3], sLo);
        const float h0 = __shfl_sync(0xffffffff, C[0], sHi);
        const float h1 = __shfl_sync(0xffffffff, C[1], sHi);
        const float h2 = __shfl_sync(0xffffffff, C[2], sHi);
        const float h3 = __shfl_sync(0xffffffff, C[3], sHi);
        A[0] = hi ? l1 : l0;    // (row g,   col t)
        A[1] = hi ? l3 : l2;    // (row g+8, col t)
        A[2] = hi ? h1 : h0;    // (row g,   col t+4)
        A[3] = hi ? h3 : h2;    // (row g+8, col t+4)
#else
        GGML_UNUSED_VARS(A, C);
        NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
    }

    // Bank-conflict swizzle (design 1.5). Element (r, c) of a staged [rows][D]
    // tile lives at r*D + (c ^ swz(r)). Legal because D is a multiple of 32, so
    // the row base contributes nothing to the bank index and the swizzle alone
    // decides it; conflict-free on both access shapes the forward uses -- the
    // B-operand read whose n is the tile's row (8 rows x 4 consecutive columns)
    // and the one whose n is the tile's column (4 rows x 8 consecutive columns) --
    // where no linear row stride is conflict-free on both.
    static __device__ __forceinline__ int swz(const int r) { return ((r & 3) << 3) | (r & 4); }

    // HOT-Step patch: flash-attn-train
    //
    // Column permutation that makes a fragment's two k-elements ADJACENT.
    // Both maps read columns {t, t+4} of an 8-wide k-slice -- a_k(l) = 4*(l/2)+t,
    // b_k(l) = 4*l+t -- and no linear layout puts those next to each other, so a
    // fragment load is four (A) or two (B) separate 4-byte loads. Storing column
    // c of each 8-group at 2*(c&3) | ((c>>2)&1) sends {t, t+4} to {2t, 2t+1} and
    // the pair becomes ONE 8-byte load: the dK/dV kernel's per-tile shared-load
    // count drops from 224 to 128.
    //
    // Composes with swz rather than replacing it: pcol stays inside its own
    // 8-column group, swz permutes within the 32-column bank block, and the pair
    // stays 8-byte aligned because pcol's low bit is (c>>2)&1 while swz's low two
    // bits are zero. Simulated over all 32 lanes of each access shape, the
    // composite is still conflict-free on PA, PB and the A-fragment read.
    //
    // Applied per KERNEL, not globally: each kernel owns its staging buffers, so
    // a kernel using pcol must use it on BOTH the store and every read, and one
    // that does not is unaffected.
    static __device__ __forceinline__ int pcol(const int c) {
        return (c & ~7) | ((c & 3) << 1) | ((c >> 2) & 1);
    }

    // The 8-byte fragment pair at (row r, k-slice ks, thread-in-group t) of a
    // pcol+swz staged tile of row stride D. Returns { column 8*ks + t,
    // column 8*ks + t + 4 } -- exactly the two k-elements every fragment wants.
    template <int D>
    static __device__ __forceinline__ float2 pair(const float * tile, const int r,
                                                  const int ks, const int t) {
        return *(const float2 *) (tile + r*D + ((8*ks + 2*t) ^ swz(r)));
    }

    // Every score in every kernel finishes HERE, so the expression is textually
    // identical by construction and cannot drift in a later edit: dQ's P and
    // dK's P must be the forward's P bit for bit or the recompute is a different
    // function (design 2). Deliberately fmaf, i.e. one rounding rather than the
    // v1 f32 path's two in `scale*dot + mv` -- the two modes are not expected to
    // agree bitwise anyway, and consistency ACROSS the three kernels is what
    // matters.
    static __device__ __forceinline__ float score(const float acc, const float scale, const float mv) {
        return fmaf(scale, acc, mv);
    }
}
