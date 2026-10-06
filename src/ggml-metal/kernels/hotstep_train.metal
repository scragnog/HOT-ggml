// HOT-Step training kernels (YuE2 / ACE-Step joint training on Metal).
// Separate per-kind library; kernels are looked up by function name.
#include "common.h"

// HOT-Step: REPEAT_BACK (docs/plans/yue2-joint-training-metal-port.md, Phase
// 3 kernel #3). Not present in upstream ggml-org/llama.cpp's Metal backend
// either (checked) -- this is new, not a port. One thread per DESTINATION
// (the smaller, post-reduction) element; each thread sums its nr0*nr1*nr2*nr3
// contributing source elements in the same i3(outermost)->i2->i1->i0(innermost)
// nesting order ggml-cpu/ops.cpp's ggml_compute_forward_repeat_back_f32 uses
// (and ggml-cuda/binbcast.cu's k_repeat_back matches too), so the summation
// order -- and therefore the bit pattern, since float addition isn't
// associative -- matches the CPU oracle exactly. Only F32 is implemented,
// matching ggml-cpu's own repeat_back (F32-only; see its outer type switch).
kernel void kernel_repeat_back_f32(
        constant ggml_metal_kargs_repeat_back & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t k1 = tgpig.x;
    const int64_t k2 = tgpig.y;
    const int64_t k3 = tgpig.z;

    const int64_t nr0 = args.ne00 / args.ne0;
    const int64_t nr1 = args.ne01 / args.ne1;
    const int64_t nr2 = args.ne02 / args.ne2;
    const int64_t nr3 = args.ne03 / args.ne3;

    device       char * dst_row = dst + k3*args.nb3 + k2*args.nb2 + k1*args.nb1;

    for (int64_t k0 = tpitg.x; k0 < args.ne0; k0 += ntg.x) {
        float sum = 0.0f;

        for (int64_t i3 = 0; i3 < nr3; ++i3) {
            const int64_t s3 = i3*args.ne3 + k3;
            for (int64_t i2 = 0; i2 < nr2; ++i2) {
                const int64_t s2 = i2*args.ne2 + k2;
                for (int64_t i1 = 0; i1 < nr1; ++i1) {
                    const int64_t s1 = i1*args.ne1 + k1;
                    device const char * src0_row = src0 + s3*args.nb03 + s2*args.nb02 + s1*args.nb01;
                    for (int64_t i0 = 0; i0 < nr0; ++i0) {
                        const int64_t s0 = i0*args.ne0 + k0;
                        sum += *((device const float *)(src0_row + s0*args.nb00));
                    }
                }
            }
        }

        *((device float *)(dst_row + k0*args.nb0)) = sum;
    }
}

// HOT-Step: RMS_NORM_BACK -- not present in upstream ggml-org/llama.cpp's
// Metal backend (checked against current master: no rms_norm_back kernel,
// dispatch case, or supports_op entry anywhere in kernels/norm.metal,
// ggml-metal-ops.cpp or ggml-metal-device.m), so written from scratch.
//
// The two-level simdgroup + threadgroup reduction here is the same shape as
// this file's own kernel_rms_norm_fuse_impl just above (the block's already
// -supported RMS_NORM forward kernel), just carrying two running sums
// (sum_xx, sum_xdz) through it instead of one. The math mirrors
// ggml-cpu/ops.cpp's ggml_compute_forward_rms_norm_back_f32:
//   sum_xx  = sum(x*x), sum_xdz = sum(x*dz)      (per row)
//   mean_eps = sum_xx/N + eps;  sum_eps = sum_xx + eps*N
//   rrms = 1/sqrt(mean_eps);  scale_x = -sum_xdz/sum_eps
//   dx[i] = (dz[i] + x[i]*scale_x) * rrms
//
// CPU accumulates sum_xx/sum_xdz in ggml_float (== double, see
// ggml-cpu/vec.h) -- Apple GPUs have no double support, so this kernel
// accumulates in float32 instead. That's not a step down from what CUDA
// already does here: ggml-cuda/norm.cu's rms_norm_back_f32 also accumulates
// sum_xx/sum_xg in plain float via warp_reduce_sum, never double. So this
// kernel's numeric test (yue2-rms-norm-back-metal-numeric-test.cpp)
// compares against the CPU (double-accumulating) reference with a
// tolerance, not bit-exact, the same way SILU_BACK's test does and for an
// analogous reason (here: reduction precision, there: transcendental ULPs).
kernel void kernel_rms_norm_back_f32(
        constant ggml_metal_kargs_rms_norm_back & args,
        device const char * src0, // dz (upstream gradient)
        device const char * src1, // x  (forward-pass input)
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg]      = 0.0f;
        shmem_f32[tiisg + 32] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const float * dz = (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const float * x  = (device const float *) (src1 + i01*args.nb11 + i02*args.nb12 + i03*args.nb13);

    float sum_xx  = 0.0f;
    float sum_xdz = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        const float xi = x[i00];
        sum_xx  += xi*xi;
        sum_xdz += xi*dz[i00];
    }
    sum_xx  = simd_sum(sum_xx);
    sum_xdz = simd_sum(sum_xdz);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg]      = sum_xx;
        shmem_f32[sgitg + 32] = sum_xdz;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum_xx  = shmem_f32[tiisg];
    sum_xx  = simd_sum(sum_xx);

    sum_xdz = shmem_f32[tiisg + 32];
    sum_xdz = simd_sum(sum_xdz);

    const float mean_eps = sum_xx/(float) args.ne00 + args.eps;
    const float sum_eps  = sum_xx + args.eps*(float) args.ne00;

    const float rrms    = 1.0f/sqrt(mean_eps);
    const float scale_x = -sum_xdz/sum_eps;

    device float * dst_row = (device float *) (dst + i01*args.nb1 + i02*args.nb2 + i03*args.nb3);
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        dst_row[i00] = (dz[i00] + x[i00]*scale_x) * rrms;
    }
}

// HOT-Step: CONVROT8 / CONVROT8_BACK.
//
// Ports tools/yue2-aitk-reference/convrot_cpu.h's scalar algorithm (already
// mirrored bit-for-bit by the CPU port in
// engine/patches/zzzz-yue2-convrot8-cpu.patch) to a threadgroup-cooperative
// GPU kernel -- one threadgroup per row. This is NOT a port of
// ggml-cuda/convrot8.cu (that path drives cuBLAS int8 GEMM); it's a
// from-scratch, correctness-first reformulation of the same scalar
// algorithm, parallelized in a way that stays bit-exact with the CPU port:
//
//   - elementwise casts/quantization: embarrassingly parallel, any thread
//     may own any index.
//   - the Hadamard rotation (self-inverse, see engine/src/convrot.h): each
//     butterfly reads/writes 4 positions no other butterfly in the same
//     stage touches, so butterflies can be freely parallelized across
//     threads; only different *stages* have a true data dependency, so a
//     barrier is required between stages, not within one.
//   - the forward per-output int8 dot product: int32 addition is exact and
//     associative regardless of order, so it's safe to give one output
//     column to one thread and let it accumulate over all `in` internally.
//   - the backward per-input gradient sum: this one is float addition
//     (NOT associative/order-independent), so -- unlike the forward dot
//     product -- each input-k's sum over `out` MUST stay a strictly
//     sequential n=0..out-1 loop within a single thread, matching the CPU
//     oracle's own loop order exactly. Only k is split across threads.
//   - amax (row max-abs, for the quantization scale): max is exact and
//     order-independent, so it uses the same two-level simdgroup+
//     threadgroup reduction trick as kernel_rms_norm_back_f32 above, with
//     0.0f (not -INFINITY) as the reduction's neutral/unused-slot value --
//     correct here (unlike a softmax-style max-reduction) because the CPU
//     oracle's own amax also starts at 0.0f and only ever compares against
//     |x| >= 0, so an empty/unused slot seeded with 0 can never win over a
//     real row value and can never suppress one either.
//
// NOTE on bit-exactness in practice: the int32 dot product and the
// sequential backward sum above are exact/order-matched as designed, but
// the Hadamard rotation's float add/sub butterflies are still genuine
// float32 arithmetic, and this project's Metal library is compiled at
// runtime via newLibraryWithSource: with MTLCompileOptions.fastMathEnabled
// left at its default true (see ggml_metal_library_init in
// ggml-metal-device.m) -- fast math permits the compiler to reassociate
// those expressions, producing ~1 ULP differences from the CPU port's
// strictly left-to-right evaluation. Confirmed empirically on Axel's M1
// Max: every case with use_bf16=false showed ~1 ULP diffs; every case with
// use_bf16=true passed exactly, since bf16's much coarser ~1/128 rounding
// step absorbs that noise. engine/tools/yue2-convrot8-metal-numeric-test.cpp
// therefore uses a (tight) tolerance rather than bit-exact comparison --
// see that file's header for the full derivation. This is a backend
// compile-option difference, not a port bug.

inline float convrot8_cast(float x, int use_bf16) {
    return use_bf16 != 0 ? bf16_round_cast<float>(x) : x;
}

// Round-half-to-even, independent of the process FP rounding mode -- bit-for-
// bit the same construction as aitk_reference::round_even / this project's
// own hs_convrot8_round_half_even (ggml-cpu port).
inline int convrot8_round_half_even(float x) {
    const float lo   = floor(x);
    const float frac = x - lo;
    int result = (int) lo;
    if (frac > 0.5f || (frac == 0.5f && (result & 1) != 0)) {
        ++result;
    }
    return result;
}

// In-place cooperative self-inverse Hadamard rotation of one length-`in` row,
// applied group-wise in groups of size `rotation` (no-op when rotation==1).
// Bit-for-bit the same radix-4 butterfly construction as
// engine/src/convrot.h's convrot_transform_group / the CPU port's
// hs_convrot8_hadamard_group, just flattened across ALL groups in the row so
// every thread in the threadgroup can help regardless of how many groups
// there are (rotation can be as small as 4 or as large as `in` itself).
//
// For a fixed stage (stride), each group contributes exactly G/4 disjoint
// butterflies (block = 4*stride, G/block blocks per group, `stride`
// butterflies per block -- (G/block)*stride == G/4, constant across all
// stages of the same group). A flattened butterfly id bid in
// [0, (in/G)*(G/4)) therefore maps 1:1 onto (group, block, off) with no
// cross-thread overlap within a stage, so only a barrier BETWEEN stages is
// required, never within one.
inline void convrot8_hadamard_row(
        threadgroup float * row,
        int64_t in,
        int64_t rotation,
        ushort  tid,
        ushort  nth) {
    if (rotation == 1) {
        return;
    }

    const int64_t g_size   = rotation;
    const int64_t n_groups = in / g_size;
    const int64_t bpg      = g_size / 4; // butterflies per group, per stage
    const int64_t total    = n_groups * bpg;

    for (int64_t stride = 1; stride < g_size; stride *= 4) {
        for (int64_t bid = tid; bid < total; bid += nth) {
            const int64_t grp   = bid / bpg;
            const int64_t local = bid % bpg;
            const int64_t bidx  = local / stride;
            const int64_t off   = local % stride;
            const int64_t base  = bidx * stride * 4;
            const int64_t p     = grp * g_size + base + off;

            const float x0 = row[p];
            const float x1 = row[p + stride];
            const float x2 = row[p + 2*stride];
            const float x3 = row[p + 3*stride];

            row[p]            =  x0 + x1 + x2 - x3;
            row[p + stride]   =  x0 + x1 - x2 + x3;
            row[p + 2*stride] =  x0 - x1 + x2 + x3;
            row[p + 3*stride] = -x0 + x1 + x2 + x3;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float norm = 1.0f / sqrt((float) g_size);
    for (int64_t k = tid; k < in; k += nth) {
        row[k] *= norm;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void kernel_convrot8_f32(
        constant ggml_metal_kargs_convrot8 & args,
        device const char * src0, // weight_i8   [in, out]  I8
        device const char * src1, // x_f32       [in, rows] F32
        device const char * src2, // scales_f32  [out]      F32
        device const char * src3, // bias_f32    [out]      F32 (dummy buffer bound when !args.has_bias)
        device       char * dst,  // out_f32     [out, rows] F32
        threadgroup char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    threadgroup float  * red     = (threadgroup float  *) shmem;
    threadgroup float  * rotated = (threadgroup float  *) (shmem + 32*sizeof(float));
    threadgroup int8_t * codes   = (threadgroup int8_t *) (shmem + 32*sizeof(float) + args.in*sizeof(float));

    if (sgitg == 0) {
        red[tiisg] = 0.0f;
    }

    const int64_t row = tgpig.x;
    const ushort  tid = tpitg.x;
    const ushort  nth = ntg.x;

    device const int8_t * weight = (device const int8_t *) src0;
    device const float  * x_row  = (device const float  *) (src1 + row*args.in*sizeof(float));
    device const float  * scales = (device const float  *) src2;
    device const float  * bias   = (device const float  *) src3;

    // 1. load + cast the activation row
    for (int64_t k = tid; k < args.in; k += nth) {
        rotated[k] = convrot8_cast(x_row[k], args.use_bf16);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. self-inverse Hadamard rotation (no-op when args.rotation == 1)
    convrot8_hadamard_row(rotated, args.in, args.rotation, tid, nth);

    // 3. unconditional second cast -- matches the oracle even when
    // args.rotation == 1 (aitk_reference::convrot8 always casts twice).
    for (int64_t k = tid; k < args.in; k += nth) {
        rotated[k] = convrot8_cast(rotated[k], args.use_bf16);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. amax = max(0, |rotated[k]|) over the whole row -- order-independent,
    // safe to reduce with the two-level simdgroup+threadgroup trick.
    float amax = 0.0f;
    for (int64_t k = tid; k < args.in; k += nth) {
        const float av = fabs(rotated[k]);
        amax = amax > av ? amax : av;
    }
    amax = simd_max(amax);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        red[sgitg] = amax;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    amax = simd_max(red[tiisg]);

    const float scale = amax > 0.0f ? amax/127.0f : 1.0f;

    // 5. quantize to int8 codes (round-half-to-even, clamp to [-127, 127])
    for (int64_t k = tid; k < args.in; k += nth) {
        int q = convrot8_round_half_even(rotated[k]/scale);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        codes[k] = (int8_t) q;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 6. per-output int8 dot product. int32 accumulation is exact regardless
    // of summation order, so one output column per thread (looping over the
    // full `in` internally) is both simple and bit-exact with the CPU port.
    device float * dst_row = (device float *) (dst + row*args.out*sizeof(float));
    for (int64_t n = tid; n < args.out; n += nth) {
        device const int8_t * wrow = weight + n*args.in;

        int32_t sum = 0;
        for (int64_t k = 0; k < args.in; ++k) {
            sum += (int32_t) codes[k] * (int32_t) wrow[k];
        }

        // Triton CUDA epilogue order: integer accumulator * (activation scale * weight scale).
        const float y = (float) sum * (scale * scales[n]);
        dst_row[n] = args.has_bias != 0
            ? convrot8_cast(y + convrot8_cast(bias[n], args.use_bf16), args.use_bf16)
            : convrot8_cast(y, args.use_bf16);
    }
}

kernel void kernel_convrot8_back_f32(
        constant ggml_metal_kargs_convrot8_back & args,
        device const char * src0, // weight_i8   [in, out]  I8
        device const char * src1, // dy_f32      [out, rows] F32
        device const char * src2, // scales_f32  [out]      F32
        device       char * dst,  // dx_f32      [in, rows] F32
        threadgroup char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    threadgroup float * grad = (threadgroup float *) shmem;

    const int64_t row = tgpig.x;
    const ushort  tid = tpitg.x;
    const ushort  nth = ntg.x;

    device const int8_t * weight = (device const int8_t *) src0;
    device const float  * dy_row = (device const float  *) (src1 + row*args.out*sizeof(float));
    device const float  * scales = (device const float  *) src2;

    // Only the activation receives a gradient (weight/scales are frozen).
    // Per input-k, sum over n of cast(dy[n]) * cast(weight[n,k]*cast(scale[n])).
    // This is float addition -- NOT associative -- so unlike the forward
    // dot product this inner loop must stay strictly sequential in n within
    // a single thread to match the CPU oracle bit-for-bit; only k is split
    // across threads.
    for (int64_t k = tid; k < args.in; k += nth) {
        float sum = 0.0f;
        for (int64_t n = 0; n < args.out; ++n) {
            const float w = convrot8_cast((float) weight[n*args.in + k] * convrot8_cast(scales[n], args.use_bf16), args.use_bf16);
            sum += convrot8_cast(dy_row[n], args.use_bf16) * w;
        }
        grad[k] = convrot8_cast(sum, args.use_bf16);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Self-inverse Hadamard un-rotation (no-op when args.rotation == 1).
    convrot8_hadamard_row(grad, args.in, args.rotation, tid, nth);

    device float * dst_row = (device float *) (dst + row*args.in*sizeof(float));
    for (int64_t k = tid; k < args.in; k += nth) {
        dst_row[k] = convrot8_cast(grad[k], args.use_bf16); // unconditional final cast -- matches the oracle
    }
}

// ============================================================================
// HOT-Step: TILED CONVROT8 / CONVROT8_BACK (opt-in: GGML_METAL_CONVROT8_TILED)
//
// Same math as kernel_convrot8_f32 / kernel_convrot8_back_f32 above, but
// with operand reuse. The per-row kernels stream the ENTIRE weight matrix
// once per activation row (a 2048x12288 gate_up weight is ~25 MB, read
// again for each of thousands of rows, the 184704x2048 LM head ~378 MB per
// row) with one thread per output doing a full-length scalar loop. These
// kernels stage tiles of both operands in threadgroup memory and let every
// staged byte serve 32-64 outputs.
//
// Numerics are identical by construction:
//
//  - Forward: the int32 dot product is computed with simdgroup float 8x8
//    MMAs on int8 values converted to float. Every product is an integer
//    with |p| <= 127*128, and every partial sum over at most
//    CR8_KCHUNK = 1024 terms is an integer with |s| <= 1024*127*128 < 2^24,
//    so each float operation is exact no matter how the hardware orders or
//    fuses it. Partials are flushed into int32 every CR8_KCHUNK and at the
//    end -- the resulting int32 sum equals the sequential int32 sum exactly.
//    Quantization (cast, Hadamard, amax, codes) is the unchanged per-row
//    code, moved into kernel_convrot8_quant_f32, which writes codes and the
//    row scale to scratch space reserved after dst
//    (ggml_metal_op_convrot8_extra). The epilogue expression is unchanged.
//
//  - Backward: float addition is not associative, so each output still
//    sums its n = 0..out-1 terms strictly in ascending order in a single
//    thread -- tiling only changes where the operands come from
//    (threadgroup memory, loaded once per tile) not the order of the adds.
//    No padded terms are ever added (the inner loop stops at `out`). The
//    Hadamard un-rotation + final cast runs as a second per-row pass
//    (kernel_convrot8_back_rot_f32) because it needs the full row.
// ============================================================================

kernel void kernel_convrot8_quant_f32(
        constant ggml_metal_kargs_convrot8 & args,
        device const char * src1,   // x_f32  [in, rows] F32
        device       char * dcodes, // codes  [in, rows] I8   (scratch)
        device       char * dscale, // scale  [rows]     F32  (scratch)
        threadgroup char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    threadgroup float * red     = (threadgroup float *) shmem;
    threadgroup float * rotated = (threadgroup float *) (shmem + 32*sizeof(float));

    if (sgitg == 0) {
        red[tiisg] = 0.0f;
    }

    const int64_t row = tgpig.x;
    const ushort  tid = tpitg.x;
    const ushort  nth = ntg.x;

    device const float * x_row = (device const float *) (src1 + row*args.in*sizeof(float));

    for (int64_t k = tid; k < args.in; k += nth) {
        rotated[k] = convrot8_cast(x_row[k], args.use_bf16);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    convrot8_hadamard_row(rotated, args.in, args.rotation, tid, nth);

    for (int64_t k = tid; k < args.in; k += nth) {
        rotated[k] = convrot8_cast(rotated[k], args.use_bf16);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float amax = 0.0f;
    for (int64_t k = tid; k < args.in; k += nth) {
        const float av = fabs(rotated[k]);
        amax = amax > av ? amax : av;
    }
    amax = simd_max(amax);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        red[sgitg] = amax;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    amax = simd_max(red[tiisg]);

    const float scale = amax > 0.0f ? amax/127.0f : 1.0f;

    device int8_t * codes = (device int8_t *) dcodes + row*args.in;
    for (int64_t k = tid; k < args.in; k += nth) {
        int q = convrot8_round_half_even(rotated[k]/scale);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        codes[k] = (int8_t) q;
    }

    if (tid == 0) {
        ((device float *) dscale)[row] = scale;
    }
}

#define CR8_NR0    64   // output columns (n) per threadgroup
#define CR8_NR1    32   // activation rows (m) per threadgroup
#define CR8_NK     32   // k per tile step
#define CR8_HLDS   40   // padded threadgroup row stride (halfs) for sw/sa
#define CR8_KCHUNK 1024 // exact-float chunk: 1024*127*128 < 2^24

// 128 threads (4 simdgroups). Simdgroup g owns output columns
// [16g, 16g+16) x all 32 rows of the tile = 4x2 blocks of 8x8.
// Threadgroup memory: 8192 bytes (half sw 64*40 + sa 32*40 = 7680 B; sc 32*64 floats aliases them).
kernel void kernel_convrot8_mm_f32(
        constant ggml_metal_kargs_convrot8 & args,
        device const char * src0,   // weight_i8 [in, out]  I8
        device const char * dcodes, // codes     [in, rows] I8  (scratch)
        device const char * dscale, // scale     [rows]     F32 (scratch)
        device const char * src2,   // scales    [out]      F32
        device const char * src3,   // bias      [out]      F32 (dummy when !has_bias)
        device       char * dst,    // out       [out, rows] F32
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    // int8 values are exact in half; half tiles halve threadgroup memory (more resident TGs)
    threadgroup half  * sw = (threadgroup half *) shmem;                                          // [NR0][HLDS]
    threadgroup half  * sa = (threadgroup half *) (shmem + CR8_NR0*CR8_HLDS*sizeof(half));        // [NR1][HLDS]
    threadgroup float * sc = (threadgroup float *) shmem; // [NR1][NR0], aliases sw/sa (only used after a barrier)

    const int r1 = tgpig.x*CR8_NR1; // first activation row of this tile
    const int r0 = tgpig.y*CR8_NR0; // first output column of this tile

    // tile loaders: weights 64 x 32 bytes (16 per thread), codes 32 x 32 bytes (8 per thread)
    const short wl_n = tiitg/2;
    const short wl_k = (tiitg%2)*16;
    const short al_m = tiitg/4;
    const short al_k = (tiitg%4)*8;

    const bool w_ok = r0 + wl_n < args.out;
    const bool a_ok = r1 + al_m < args.rows;

    device const char4 * wp = (device const char4 *) (src0   + (int64_t) (w_ok ? r0 + wl_n : 0)*args.in + wl_k);
    device const char4 * ap = (device const char4 *) (dcodes + (int64_t) (a_ok ? r1 + al_m : 0)*args.in + al_k);

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    int32_t acc[16];
    for (short j = 0; j < 16; ++j) {
        acc[j] = 0;
    }

    for (int k0 = 0; k0 < args.in; k0 += CR8_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 4; ++i) {
            const char4 v = w_ok ? wp[i] : char4(0);
            *(threadgroup half4 *) (sw + wl_n*CR8_HLDS + wl_k + 4*i) = half4(v);
        }
        for (short i = 0; i < 2; ++i) {
            const char4 v = a_ok ? ap[i] : char4(0);
            *(threadgroup half4 *) (sa + al_m*CR8_HLDS + al_k + 4*i) = half4(v);
        }
        wp += CR8_NK/4;
        ap += CR8_NK/4;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short kk = 0; kk < CR8_NK; kk += 8) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mw[2];

            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], sa + (8*i)*CR8_HLDS + kk, CR8_HLDS);
            }
            for (short i = 0; i < 2; ++i) {
                // transposed load: [n][k] tile -> (k x n) block
                simdgroup_load(mw[i], sw + (16*sgitg + 8*i)*CR8_HLDS + kk, CR8_HLDS, 0, true);
            }
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], ma[i/2], mw[i%2], mc[i]);
            }
        }

        // flush the (exact, integer-valued) float partials into int32
        const int kn = k0 + CR8_NK;
        if (kn % CR8_KCHUNK == 0 || kn >= args.in) {
            threadgroup_barrier(mem_flags::mem_threadgroup); // all simdgroups done reading sw/sa (sc aliases sw)
            for (short i = 0; i < 8; ++i) {
                simdgroup_store(mc[i], sc + (8*(i/2))*CR8_NR0 + 16*sgitg + 8*(i%2), CR8_NR0);
                mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short j = 0; j < 16; ++j) {
                acc[j] += (int32_t) sc[tiitg + 128*j];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    device const float * ascale = (device const float *) dscale;
    device const float * scales = (device const float *) src2;
    device const float * bias   = (device const float *) src3;
    device       float * out    = (device       float *) dst;

    for (short j = 0; j < 16; ++j) {
        const int idx = tiitg + 128*j;
        const int row = r1 + idx/CR8_NR0;
        const int col = r0 + idx%CR8_NR0;
        if (row < args.rows && col < args.out) {
            // Triton CUDA epilogue order, unchanged from kernel_convrot8_f32.
            const float y = (float) acc[j] * (ascale[row] * scales[col]);
            out[(int64_t) row*args.out + col] = args.has_bias != 0
                ? convrot8_cast(y + convrot8_cast(bias[col], args.use_bf16), args.use_bf16)
                : convrot8_cast(y, args.use_bf16);
        }
    }
}

#define CR8B_BM 64 // rows per threadgroup (weight dequant is amortized over BM rows)
#define CR8B_BK 64 // input columns (k) per threadgroup
#define CR8B_BN 32 // reduction (n) step

// 128 threads; thread (tm = tid/16, tk = tid%16) owns rows r1 + tm + 8*i
// (i = 0..7) x columns k0 + 4*tk + j (j = 0..3). Threadgroup memory:
// sdy 64*32 + sw 32*64 floats = 16384 bytes.
kernel void kernel_convrot8_back_mm_f32(
        constant ggml_metal_kargs_convrot8_back & args,
        device const char * src0, // weight_i8 [in, out]  I8
        device const char * src1, // dy_f32    [out, rows] F32
        device const char * src2, // scales    [out]       F32
        device       char * dst,  // dx        [in, rows]  F32 (pre-rotation when rotation != 1)
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    threadgroup float * sdy = (threadgroup float *) shmem;                                  // [BM][BN]
    threadgroup float * sw  = (threadgroup float *) (shmem + CR8B_BM*CR8B_BN*sizeof(float)); // [BN][BK]

    const int r1 = tgpig.x*CR8B_BM;
    const int k0 = tgpig.y*CR8B_BK;

    const short tk = tiitg%16;
    const short tm = tiitg/16;

    const short dl_m = tiitg/2;       // 0..63
    const short dl_n = (tiitg%2)*16;  // 16 consecutive n per thread
    const short wl_n = tiitg/4;
    const short wl_k = (tiitg%4)*16;

    device const int8_t * w  = (device const int8_t *) src0;
    device const float  * dy = (device const float  *) src1;
    device const float  * sc = (device const float  *) src2;

    // vector loads need 16-byte (dy) / 4-byte (w) aligned rows
    const bool dy_vec = (args.out % 4) == 0;
    const bool w_vec  = (args.in  % 4) == 0;

    float acc[8][4];
    for (short i = 0; i < 8; ++i) {
        for (short j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int n0 = 0; n0 < args.out; n0 += CR8B_BN) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const int row = r1 + dl_m;
            if (dy_vec && row < args.rows && n0 + dl_n + 16 <= args.out) {
                // vector path: 16 consecutive n, same per-element cast
                device const float4 * src = (device const float4 *) (dy + (int64_t) row*args.out + n0 + dl_n);
                for (short e = 0; e < 4; ++e) {
                    const float4 v = src[e];
                    *(threadgroup float4 *) (sdy + dl_m*CR8B_BN + dl_n + 4*e) = float4(
                        convrot8_cast(v[0], args.use_bf16), convrot8_cast(v[1], args.use_bf16),
                        convrot8_cast(v[2], args.use_bf16), convrot8_cast(v[3], args.use_bf16));
                }
            } else {
                for (short e = 0; e < 16; ++e) {
                    const int n = n0 + dl_n + e;
                    sdy[dl_m*CR8B_BN + dl_n + e] = (row < args.rows && n < args.out)
                        ? convrot8_cast(dy[(int64_t) row*args.out + n], args.use_bf16)
                        : 0.0f;
                }
            }
        }
        {
            const int  n    = n0 + wl_n;
            const bool n_ok = n < args.out;
            const float s   = n_ok ? convrot8_cast(sc[n], args.use_bf16) : 0.0f;
            if (w_vec && n_ok && k0 + wl_k + 16 <= args.in) {
                // vector path: 16 consecutive k as 4 x char4, same expression per element
                device const char4 * src = (device const char4 *) (w + (int64_t) n*args.in + k0 + wl_k);
                for (short e = 0; e < 4; ++e) {
                    const char4 v = src[e];
                    *(threadgroup float4 *) (sw + wl_n*CR8B_BK + wl_k + 4*e) = float4(
                        convrot8_cast((float) v[0] * s, args.use_bf16), convrot8_cast((float) v[1] * s, args.use_bf16),
                        convrot8_cast((float) v[2] * s, args.use_bf16), convrot8_cast((float) v[3] * s, args.use_bf16));
                }
            } else {
                for (short e = 0; e < 16; ++e) {
                    const int k = k0 + wl_k + e;
                    // same expression as kernel_convrot8_back_f32's `w`
                    sw[wl_n*CR8B_BK + wl_k + e] = (n_ok && k < args.in)
                        ? convrot8_cast((float) w[(int64_t) n*args.in + k] * s, args.use_bf16)
                        : 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // exact term count: never adds a padded term past `out`
        const int nn_end = min(CR8B_BN, args.out - n0);
        for (int nn = 0; nn < nn_end; ++nn) {
            const float4 wv = *(threadgroup const float4 *) (sw + nn*CR8B_BK + 4*tk);
            for (short i = 0; i < 8; ++i) {
                const float a = sdy[(tm + 8*i)*CR8B_BN + nn];
                acc[i][0] += a * wv[0];
                acc[i][1] += a * wv[1];
                acc[i][2] += a * wv[2];
                acc[i][3] += a * wv[3];
            }
        }
    }

    device float * dx = (device float *) dst;
    for (short i = 0; i < 8; ++i) {
        const int row = r1 + tm + 8*i;
        if (row >= args.rows) {
            continue;
        }
        for (short j = 0; j < 4; ++j) {
            const int k = k0 + 4*tk + j;
            if (k < args.in) {
                const float g = convrot8_cast(acc[i][j], args.use_bf16);
                // rotation == 1: Hadamard is a no-op, apply the final cast here
                dx[(int64_t) row*args.in + k] = args.rotation == 1 ? convrot8_cast(g, args.use_bf16) : g;
            }
        }
    }
}

// convrot8 backward with half simdgroup MMA and forward-style tiles (exact, range-scaled; see kernel_convrot8_back_mm_hs_f32).
// dx[r][k] = sum_n dy[r][n] * (q[n][k] * s[n]). Tile 32 rows x 64 k, NK = 32 n per step,
// 128 threads (4 simdgroups, each 32 rows x 16 k), 8 KB threadgroup memory.
// Requires in % 64 == 0 and out % 32 == 0 (the host checks).
#define CR8H_NR1 32
#define CR8H_NR0 64
#define CR8H_NK  32
#define CR8H_LDA 40
#define CR8H_LDB 72

inline float4 cr8h_cast4(float4 x, int use_bf16) {
    return float4(convrot8_cast(x[0], use_bf16), convrot8_cast(x[1], use_bf16),
                  convrot8_cast(x[2], use_bf16), convrot8_cast(x[3], use_bf16));
}

// Debug (GGML_METAL_CR8B_CHECK=1): compare two dx buffers bit for bit, bf16-ulp histogram.
kernel void kernel_convrot8_back_cmp_f32(
        constant ggml_metal_kargs_convrot8_back & args [[buffer(0)]],
        device const float * a  [[buffer(1)]],
        device const float * b  [[buffer(2)]],
        device atomic_uint * st [[buffer(3)]],
        constant int & slot     [[buffer(4)]],
        uint  gid  [[thread_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const int64_t n = (int64_t) args.rows*args.in;
    const bool valid = (int64_t) gid < n;
    const float x = valid ? a[gid] : 0.0f;
    const float y = valid ? b[gid] : 0.0f;
    const uint ba = as_type<uint>(x);
    const uint bb = as_type<uint>(y);
    const bool mis = valid && ba != bb;
    const bool same = (ba >> 31) == (bb >> 31);
    const int  d = same ? abs((int) (ba >> 16) - (int) (bb >> 16)) : 99;
    const uint m  = simd_sum(mis ? 1u : 0u);
    const uint c1 = simd_sum((mis && d == 1) ? 1u : 0u);
    const uint cg = simd_sum((mis && d > 1) ? 1u : 0u);
    const float ad = valid ? abs(x - y) : 0.0f;
    const float ar = valid ? abs(y) : 0.0f;
    const float md = simd_max(ad);
    const float mr = simd_max(ar);
    if (lane == 0) {
        device atomic_uint * s = st + slot*8;
        atomic_fetch_add_explicit(s + 0, m,  memory_order_relaxed);
        atomic_fetch_add_explicit(s + 1, c1, memory_order_relaxed);
        atomic_fetch_add_explicit(s + 2, cg, memory_order_relaxed);
        atomic_fetch_max_explicit(s + 3, as_type<uint>(md), memory_order_relaxed);
        atomic_fetch_max_explicit(s + 4, as_type<uint>(mr), memory_order_relaxed);
    }
}

// Row exponent pre-pass for kernel_convrot8_back_mm_hs_f32: one simdgroup per dy row.
kernel void kernel_convrot8_back_rowexp_f32(
        constant ggml_metal_kargs_convrot8_back & args,
        device const char * src1,   // dy_f32 [out, rows]
        device       char * dst,    // float2 [rows]
        uint   row  [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    device const float4 * rp = (device const float4 *) (src1 + (int64_t) row*args.out*sizeof(float));
    const int n4 = args.out/4;
    float4 am4 = float4(0.0f);
    for (int c = lane; c < n4; c += 256) {
        #pragma unroll
        for (short u = 0; u < 8; ++u) {
            const int ci = c + 32*u;
            if (ci < n4) {
                am4 = max(am4, abs(rp[ci]));
            }
        }
    }
    float amax = max(max(am4.x, am4.y), max(am4.z, am4.w));
    amax = simd_max(amax);
    if (lane == 0) {
        int e = 0;
        if (amax > 0.0f && isfinite(amax)) {
            int ex;
            frexp(amax, ex);
            e = 13 - ex;
        }
        ((device float2 *) dst)[row] = float2(ldexp(1.0f, e), ldexp(1.0f, -(e + args.wexp)));
    }
}

// Exact half-MMA backward with range scaling (default for in % 64 == 0 && out % 32 == 0; GGML_METAL_CR8B=0 disables).
// dy row r is scaled by 2^e_r (chosen from the row's absmax so the scaled values stay in half range),
// weights by 2^wexp (host-chosen per matrix, host guarantees half-exactness or falls back).
// All scalings are powers of two, applied before the exact bf16 roundings' products and undone in float.
kernel void kernel_convrot8_back_mm_hs_f32(
        constant ggml_metal_kargs_convrot8_back & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        device const float2 * scr,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half  * sA = (threadgroup half *) shmem;
    threadgroup half  * sB = (threadgroup half *) (shmem + CR8H_NR1*CR8H_LDA*sizeof(half));
    threadgroup float * sC = (threadgroup float *) shmem;                 // [16][NR0] used per phase
    threadgroup float * sFw  = (threadgroup float *) (shmem + 7936);      // [32] 2^e_r
    threadgroup float * sInv = (threadgroup float *) (shmem + 7936 + 128); // [32] 2^-(e_r+wexp)

    const int r1 = tgpig.x*CR8H_NR1;
    const int k0 = tgpig.y*CR8H_NR0;

    const short al_m = tiitg/4;
    const short al_n = (tiitg%4)*8;
    const short wl_n = tiitg/4;
    const short wl_k = (tiitg%4)*16;

    device const int8_t * w   = (device const int8_t *) src0;
    device const float  * dy  = (device const float  *) src1;
    device const float  * scl = (device const float  *) src2;

    const int  arow = r1 + al_m;
    const bool a_ok = arow < args.rows;

    // per-row scale factors come from kernel_convrot8_back_rowexp_f32: float2 {2^e_r, 2^-(e_r+wexp)}
    if (tiitg < CR8H_NR1) {
        const int row = r1 + tiitg;
        const float2 sc = row < args.rows ? scr[row] : float2(1.0f);
        sFw[tiitg]  = sc.x;
        sInv[tiitg] = sc.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rscale = sFw[al_m];
    const float wmul   = ldexp(1.0f, args.wexp);

    device const float4 * dyp = (device const float4 *) (dy + (int64_t) (a_ok ? arow : 0)*args.out + al_n);

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int n0 = 0; n0 < args.out; n0 += CR8H_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const float4 v0 = a_ok ? dyp[0] : float4(0.0f);
            const float4 v1 = a_ok ? dyp[1] : float4(0.0f);
            *(threadgroup half4 *) (sA + al_m*CR8H_LDA + al_n)     = half4(cr8h_cast4(v0, args.use_bf16)*rscale);
            *(threadgroup half4 *) (sA + al_m*CR8H_LDA + al_n + 4) = half4(cr8h_cast4(v1, args.use_bf16)*rscale);
            dyp += CR8H_NK/4;
        }
        {
            const int   n = n0 + wl_n;
            const float s = convrot8_cast(scl[n], args.use_bf16);
            device const char4 * src = (device const char4 *) (w + (int64_t) n*args.in + k0 + wl_k);
            for (short e = 0; e < 4; ++e) {
                const char4 v = src[e];
                *(threadgroup half4 *) (sB + wl_n*CR8H_LDB + wl_k + 4*e) = half4(cr8h_cast4(float4(v)*s, args.use_bf16)*wmul);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short kk = 0; kk < CR8H_NK; kk += 8) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];

            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], sA + (8*i)*CR8H_LDA + kk, CR8H_LDA);
            }
            for (short j = 0; j < 2; ++j) {
                simdgroup_load(mb[j], sB + kk*CR8H_LDB + 16*sgitg + 8*j, CR8H_LDB);
            }
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], ma[i/2], mb[i%2], mc[i]);
            }
        }
    }

    device float * dx = (device float *) dst;
    for (short ph = 0; ph < 2; ++ph) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 4; ++i) {
            simdgroup_store(mc[4*ph + i], sC + (8*(i/2))*CR8H_NR0 + 16*sgitg + 8*(i%2), CR8H_NR0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short j = 0; j < 8; ++j) {
            const int idx = tiitg + 128*j;
            const int rl  = 16*ph + idx/CR8H_NR0;
            const int row = r1 + rl;
            const int k   = k0 + idx%CR8H_NR0;
            if (row < args.rows) {
                const float g = convrot8_cast(sC[idx]*sInv[rl], args.use_bf16);
                dx[(int64_t) row*args.in + k] = args.rotation == 1 ? convrot8_cast(g, args.use_bf16) : g;
            }
        }
    }
}

// Per row: Hadamard un-rotation + unconditional final cast of the
// pre-rotation gradient written by kernel_convrot8_back_mm_f32 (in place).
kernel void kernel_convrot8_back_rot_f32(
        constant ggml_metal_kargs_convrot8_back & args,
        device       char * dst,  // dx [in, rows] F32, in place
        threadgroup char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    threadgroup float * grad = (threadgroup float *) shmem;

    const int64_t row = tgpig.x;
    const ushort  tid = tpitg.x;
    const ushort  nth = ntg.x;

    device float * dst_row = (device float *) (dst + row*args.in*sizeof(float));

    for (int64_t k = tid; k < args.in; k += nth) {
        grad[k] = dst_row[k];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    convrot8_hadamard_row(grad, args.in, args.rotation, tid, nth);

    for (int64_t k = tid; k < args.in; k += nth) {
        dst_row[k] = convrot8_cast(grad[k], args.use_bf16);
    }
}

// HOT-Step: OUT_PROD (kernel #6). F32-only -- see ggml_metal_kargs_out_prod's
// own comment in ggml-metal-impl.h for why the CPU reference's quantized/F16
// src0 variants aren't needed here.
//
// dst[i0,i1,i2,i3] = sum_{k=0}^{ne01-1} src0[i0,k,i02,i03] * src1[i1,k,i2,i3]
// (i02 = i2/dps2, i03 = i3/dps3 -- GQA-style broadcast of src0 over dst's
// higher dims, mirroring ggml-cpu/ops.cpp's ggml_compute_forward_out_prod_f32
// exactly, including its accumulation order: for a fixed dst element, CPU's
// own blocked/tiled loop still visits k purely ascending 0..ne01-1, which is
// what this kernel matches by giving one thread the whole per-(i0,row)
// reduction sequentially, instead of splitting the reduction itself across
// threads/simdgroups (that would reorder float addition, unlike the plain
// per-thread max/int32-sum tricks used elsewhere in this file).
//
// One threadgroup per dst "row" (i1,i2,i3); each thread owns one or more i0
// columns and does its own sequential k-loop -- no threadgroup memory, no
// cross-thread reduction, so (unlike RMS_NORM_BACK/CONVROT8) there's no
// nth-must-be-a-multiple-of-32 constraint here.
kernel void kernel_out_prod_f32(
        constant ggml_metal_kargs_out_prod & args,
        device const char * src0, // [ne00, ne01, ne02, ne03] F32
        device const char * src1, // [ne1,  ne01, ne2,  ne3 ] F32 (ne1/ne2/ne3 implied by dispatch grid)
        device       char * dst,  // [ne00, ne1,  ne2,  ne3 ] F32
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t i1 = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t i02 = i2 / args.dps2;
    const int64_t i03 = i3 / args.dps3;

    device const char * src0_base = src0 +                    i02*args.nb02 + i03*args.nb03;
    device const char * src1_row  = src1 + i1*args.nb10 +      i2*args.nb12  + i3*args.nb13;
    device       char * dst_row   = dst  + i1*args.nb1  +      i2*args.nb2   + i3*args.nb3;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        float sum = 0.0f;
        for (int64_t k = 0; k < args.ne01; ++k) {
            device const float * s0 = (device const float *) (src0_base + i0*args.nb00 + k*args.nb01);
            device const float * s1 = (device const float *) (src1_row  +               k*args.nb11);
            sum += (*s0) * (*s1);
        }
        device float * d = (device float *) (dst_row + i0*sizeof(float));
        *d = sum;
    }
}

// ============================================================================
// HOT-Step: TILED OUT_PROD (opt-in: GGML_METAL_OUT_PROD_TILED)
//
// The per-row kernel above gives ONE THREAD the entire reduction over
// `ne01` for a single (i0,i1) output element (see its own comment for why:
// float addition is not associative, so bit-exactness against the CPU
// reference requires a strictly sequential accumulation, matched here
// order-for-order). For this project's LoRA "B"-adapter gradient sites
// (yue2_aitk_graph::linear()'s second mul_mat, `delta = adapter->b @ ax`)
// that reduction (ne01 = the adapter's fused output width, up to 12288 for
// gate_up) runs on only `ne00` threads (the adapter rank, e.g. 32) -- one
// simdgroup, each thread scanning thousands of elements alone.
//
// This kernel trades that bit-exactness for a real speedup: a standard
// tiled GEMM using simdgroup_float8x8 (genuine float32 accumulation in
// registers throughout -- NOT the half-precision path this hardware's
// non-tensor-API kernel_mul_mm falls back to for F32xF32, see
// engine/patches/mm-backward.patch's own README entry and
// ggml-metal.metal's kernel_mul_mm template instantiation for
// "kernel_mul_mm_f32_f32": SA/SB = half on Apple7/non-tensor devices).
// simdgroup_float8x8's MMA reduces in a blocked/pairwise order, not
// ggml_compute_forward_out_prod_f32's ascending k=0..ne01-1, so results
// will differ from the CPU oracle by ordinary float32 rounding noise, not
// bit-for-bit -- validated with a TOLERANCE by
// yue2-out-prod-metal-tiled-test.cpp, not the bit-exact
// yue2-out-prod-metal-numeric-test.cpp (which continues to gate the
// default, untiled kernel above).
//
// Fully general over src0/src1 strides (no contiguity assumed, matching
// kernel_out_prod_f32's own approach) and over partial M/N tiles; GQA-style
// broadcast (dps2/dps3) is resolved once per threadgroup exactly as above.
// ============================================================================

#define OP_NR0 64 // M tile (ne00 / dst rows)
#define OP_NR1 32 // N tile (ne1  / dst cols)
#define OP_NK  32 // K tile step (ne01, the reduction dim)
#define OP_LDS 36 // padded threadgroup row stride (floats), avoids bank conflicts

// 128 threads (4 simdgroups). Simdgroup g owns M-columns [16g, 16g+16) of
// the 64-wide M tile, across the whole 32-wide N tile -- same partition as
// kernel_convrot8_mm_f32, just without its int8/quantization machinery:
// operands are already float32, no scale/bias epilogue, and no periodic
// int32-flush trick is needed (nothing here requires exact accumulation).
//
// Threadgroup memory: sA (src0/M tile) 64*36 + sB (src1/N tile) 32*36 +
// sc (fp32 store scratch) 32*64 floats = 22016 bytes.
kernel void kernel_out_prod_mm_f32(
        constant ggml_metal_kargs_out_prod & args,
        device const char * src0, // [ne00, ne01, ne02, ne03] F32, general strides
        device const char * src1, // accessed [i1,k] via nb10/nb11 (+ nb12/nb13), general strides
        device       char * dst,  // [ne00, ne1, ne2, ne3] F32
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup float * sA = (threadgroup float *) shmem;                                    // [OP_NR0][OP_LDS]
    threadgroup float * sB = (threadgroup float *) (shmem + OP_NR0*OP_LDS*sizeof(float));     // [OP_NR1][OP_LDS]
    threadgroup float * sc = (threadgroup float *) (shmem + (OP_NR0 + OP_NR1)*OP_LDS*sizeof(float)); // [OP_NR1][OP_NR0]

    const int r1 = int(tgpig.x)*OP_NR1; // first N (i1) index of this tile
    const int r0 = int(tgpig.y)*OP_NR0; // first M (i0) index of this tile

    // z folds (i2,i3): z = i3*args.ne2 + i2 (matches the (rows,ne2,ne3)-as-(x,y,z)
    // convention of the per-row kernel's own dispatch, just with x/y now tiled).
    const int i2  = int(tgpig.z) % args.ne2;
    const int i3  = int(tgpig.z) / args.ne2;
    const int i02 = i2 / args.dps2;
    const int i03 = i3 / args.dps3;

    device const char * src0_base = src0 + (int64_t) i02*args.nb02 + (int64_t) i03*args.nb03;
    device const char * src1_base = src1 + (int64_t) i2*args.nb12  + (int64_t) i3*args.nb13;

    // tile loaders: sA 64x32 (16 scalars/thread), sB 32x32 (8 scalars/thread).
    // General strides (not assumed contiguous), so scalar loads throughout --
    // matches kernel_out_prod_f32's own per-element addressing.
    const short al_n = tiitg/2;        // 0..63, sA row (i0) this thread loads
    const short al_k0 = (tiitg%2)*16;  // 0 or 16, first k this thread loads
    const short bl_n = tiitg/4;        // 0..31, sB row (i1) this thread loads
    const short bl_k0 = (tiitg%4)*8;   // 0,8,16,24, first k this thread loads

    const bool a_row_ok = r0 + al_n < args.ne00;
    const bool b_row_ok = r1 + bl_n < args.ne1;

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int k0 = 0; k0 < args.ne01; k0 += OP_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short e = 0; e < 16; ++e) {
            const int k = k0 + al_k0 + e;
            sA[al_n*OP_LDS + al_k0 + e] = (a_row_ok && k < args.ne01)
                ? *(device const float *) (src0_base + (int64_t) (r0 + al_n)*args.nb00 + (int64_t) k*args.nb01)
                : 0.0f;
        }
        for (short e = 0; e < 8; ++e) {
            const int k = k0 + bl_k0 + e;
            sB[bl_n*OP_LDS + bl_k0 + e] = (b_row_ok && k < args.ne01)
                ? *(device const float *) (src1_base + (int64_t) (r1 + bl_n)*args.nb10 + (int64_t) k*args.nb11)
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int kk_end = min(OP_NK, args.ne01 - k0);
        for (short kk = 0; kk < kk_end; kk += 8) {
            simdgroup_float8x8 ma[4]; // from sB (N dim, i1) -- 32 rows in 4 blocks of 8
            simdgroup_float8x8 mw[2]; // from sA (M dim, i0) -- this simdgroup's 16 cols in 2 blocks of 8

            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], sB + (8*i)*OP_LDS + kk, OP_LDS);
            }
            for (short i = 0; i < 2; ++i) {
                // transpose_matrix=true: sA is stored [i0][k] (row=i0), but
                // the B operand of ma@mw needs [k][i0] (row=k) -- same fix
                // kernel_convrot8_mm_f32's mw load already applies.
                simdgroup_load(mw[i], sA + (16*sgitg + 8*i)*OP_LDS + kk, OP_LDS, 0, true);
            }
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], ma[i/2], mw[i%2], mc[i]);
            }
        }
    }

    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], sc + (8*(i/2))*OP_NR0 + 16*sgitg + 8*(i%2), OP_NR0);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device char * dst_base = dst + (int64_t) i2*args.nb2 + (int64_t) i3*args.nb3;

    // Bounds-checked scalar writeback (partial tiles at the M/N edges are
    // the common case for this op's real shapes -- e.g. rank=32 as the M
    // dimension in the "B"-adapter gradient -- so there's no "always full
    // tile" fast path worth a second code path here).
    for (short j = tiitg; j < OP_NR1*OP_NR0; j += 128) {
        const short row = j / OP_NR0; // i1 offset within tile
        const short col = j % OP_NR0; // i0 offset within tile
        const int i1 = r1 + row;
        const int i0 = r0 + col;
        if (i1 < args.ne1 && i0 < args.ne00) {
            *(device float *) (dst_base + (int64_t) i1*args.nb1 + (int64_t) i0*sizeof(float)) = sc[row*OP_NR0 + col];
        }
    }
}

// HOT-Step: small-M variant of kernel_out_prod_mm_f32 above (same opt-in
// switch, GGML_METAL_OUT_PROD_TILED). For M (ne00, e.g. this project's LoRA
// adapter rank, 32) <= OP_NR0_SM, the kernel above wastes half its
// threadgroup: its M tile is fixed at 64, so with a real M of 32,
// simdgroups 2-3 (whose 16-wide M slice starts at 32/48) run their whole
// k-loop with a_row_ok false every iteration -- pure masked-zero work.
//
// A first attempt at fixing this (since reverted) shrank the M tile to 32
// but *doubled* the N tile to 64 to keep 4 full simdgroups busy -- that
// halved the threadgroup count for this project's real shapes (e.g. 32->16
// threadgroups at n=1024 on a GPU with far fewer than 32 cores), and the
// resulting occupancy loss measured *slower* than the kernel above, not
// faster (confirmed by an explicit A/B rerun with the same kernel and
// tile-shape held fixed except for that one change).
//
// This version keeps BOTH tile dimensions matched to the actual problem
// instead: M tile shrinks to 32 (an exact fit for a rank-32 adapter, so
// a_row_ok is never false here for this project's own sites) and N tile
// STAYS at 32 (unchanged from the kernel above), so the threadgroup grid
// -- and therefore occupancy -- is identical to the kernel above. What
// changes is how the 4 simdgroups split the now-32x32 tile: a 2x2 grid
// (2 simdgroups split N into halves, 2 split M into halves) instead of
// the kernel above's 1x4 split of a 64-wide M axis alone, so every
// simdgroup does useful work on every iteration. Threadgroup memory drops
// to 32*36 + 32*36 + 32*32 = 13312 bytes (vs the 22016 bytes above), a
// bonus, not the point of the change.
//
// Selected by the dispatch side (ggml_metal_op_out_prod) only when
// ne00 <= OP_NR0_SM; kernel_out_prod_mm_f32 above is unchanged and still
// used for every larger-M case (including this project's own "A"-adapter
// gradient sites, M in the thousands).
#define OP_NR0_SM 32 // M tile (small-M variant): exact fit for a rank-32 adapter
#define OP_NR1_SM 32 // N tile (small-M variant): UNCHANGED from the kernel above --
                      // threadgroup count (occupancy) must not regress.

kernel void kernel_out_prod_mm_f32_sm(
        constant ggml_metal_kargs_out_prod & args,
        device const char * src0, // [ne00, ne01, ne02, ne03] F32, general strides
        device const char * src1, // accessed [i1,k] via nb10/nb11 (+ nb12/nb13), general strides
        device       char * dst,  // [ne00, ne1, ne2, ne3] F32
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup float * sA = (threadgroup float *) shmem;                                       // [OP_NR0_SM][OP_LDS]
    threadgroup float * sB = (threadgroup float *) (shmem + OP_NR0_SM*OP_LDS*sizeof(float));     // [OP_NR1_SM][OP_LDS]
    threadgroup float * sc = (threadgroup float *) (shmem + (OP_NR0_SM + OP_NR1_SM)*OP_LDS*sizeof(float)); // [OP_NR1_SM][OP_NR0_SM]

    const int r1 = int(tgpig.x)*OP_NR1_SM; // first N (i1) index of this tile
    const int r0 = int(tgpig.y)*OP_NR0_SM; // first M (i0) index of this tile

    const int i2  = int(tgpig.z) % args.ne2;
    const int i3  = int(tgpig.z) / args.ne2;
    const int i02 = i2 / args.dps2;
    const int i03 = i3 / args.dps3;

    device const char * src0_base = src0 + (int64_t) i02*args.nb02 + (int64_t) i03*args.nb03;
    device const char * src1_base = src1 + (int64_t) i2*args.nb12  + (int64_t) i3*args.nb13;

    // tile loaders: sA and sB are both 32x32 now (8 scalars/thread each) --
    // symmetric, unlike the kernel above's 64x32/32x32 split.
    const short al_n = tiitg/4;        // 0..31, sA row (i0) this thread loads
    const short al_k0 = (tiitg%4)*8;   // 0,8,16,24, first k this thread loads
    const short bl_n = tiitg/4;        // 0..31, sB row (i1) this thread loads
    const short bl_k0 = (tiitg%4)*8;   // 0,8,16,24, first k this thread loads

    const bool a_row_ok = r0 + al_n < args.ne00;
    const bool b_row_ok = r1 + bl_n < args.ne1;

    // 2x2 simdgroup split of the 32x32 tile: sg_n picks this simdgroup's
    // 16-wide N half (of sB), sg_m picks its 16-wide M half (of sA) --
    // every simdgroup covers a distinct, real quadrant, none of it padding.
    const short sg_n = sgitg / 2;
    const short sg_m = sgitg % 2;

    simdgroup_float8x8 mc[4];
    for (short i = 0; i < 4; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int k0 = 0; k0 < args.ne01; k0 += OP_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short e = 0; e < 8; ++e) {
            const int k = k0 + al_k0 + e;
            sA[al_n*OP_LDS + al_k0 + e] = (a_row_ok && k < args.ne01)
                ? *(device const float *) (src0_base + (int64_t) (r0 + al_n)*args.nb00 + (int64_t) k*args.nb01)
                : 0.0f;
        }
        for (short e = 0; e < 8; ++e) {
            const int k = k0 + bl_k0 + e;
            sB[bl_n*OP_LDS + bl_k0 + e] = (b_row_ok && k < args.ne01)
                ? *(device const float *) (src1_base + (int64_t) (r1 + bl_n)*args.nb10 + (int64_t) k*args.nb11)
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int kk_end = min(OP_NK, args.ne01 - k0);
        for (short kk = 0; kk < kk_end; kk += 8) {
            simdgroup_float8x8 ma[2]; // from sB (N dim, i1) -- this simdgroup's 16-wide N half, 2 blocks of 8
            simdgroup_float8x8 mw[2]; // from sA (M dim, i0) -- this simdgroup's 16-wide M half, 2 blocks of 8

            for (short i = 0; i < 2; ++i) {
                simdgroup_load(ma[i], sB + (16*sg_n + 8*i)*OP_LDS + kk, OP_LDS);
            }
            for (short i = 0; i < 2; ++i) {
                // transpose_matrix=true: sA is stored [i0][k] (row=i0), but
                // the B operand of ma@mw needs [k][i0] (row=k) -- same as
                // kernel_out_prod_mm_f32's mw load above.
                simdgroup_load(mw[i], sA + (16*sg_m + 8*i)*OP_LDS + kk, OP_LDS, 0, true);
            }
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc[i], ma[i/2], mw[i%2], mc[i]);
            }
        }
    }

    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc[i], sc + (16*sg_n + 8*(i/2))*OP_NR0_SM + (16*sg_m + 8*(i%2)), OP_NR0_SM);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device char * dst_base = dst + (int64_t) i2*args.nb2 + (int64_t) i3*args.nb3;

    // Bounds-checked scalar writeback, same rationale as
    // kernel_out_prod_mm_f32's own writeback loop.
    for (short j = tiitg; j < OP_NR1_SM*OP_NR0_SM; j += 128) {
        const short row = j / OP_NR0_SM; // i1 offset within tile
        const short col = j % OP_NR0_SM; // i0 offset within tile
        const int i1 = r1 + row;
        const int i0 = r0 + col;
        if (i1 < args.ne1 && i0 < args.ne00) {
            *(device float *) (dst_base + (int64_t) i1*args.nb1 + (int64_t) i0*sizeof(float)) = sc[row*OP_NR0_SM + col];
        }
    }
}

// HOT-Step: FLASH_ATTN_TRAIN / FLASH_ATTN_TRAIN_BACK (kernel #7). See
// ggml_metal_kargs_flash_attn_train's own comment in ggml-metal-impl.h for
// the design (one thread per query/kv row, two passes, tolerance-gated
// against the CPU oracle -- not bit-exact by construction, unlike
// REPEAT_BACK/OUT_PROD's plain accumulator loops, because this kernel's
// online-softmax scan is grouped differently than the CPU's BQ/BK tiling and
// both sides call exp/log, whose GPU and CPU libm implementations are not
// required to agree to the last bit).
//
// Deliberate deviation from ggml_soft_max_ext, carried over from the CPU
// oracle (see ggml-cpu/ops.cpp's own comment): a query row whose every key
// is masked yields O = 0, LSE = 0 -- never NaN, since the packed tensor's
// gradient is ggml_scale(packed, 0.0f) and 0*NaN is NaN.
//
// All three kernels detect masked keys and seed the softmax recurrence with
// literal -INFINITY comparisons/values. This is deliberate and known to be
// fast-math-dependent (this library compiles with MTLCompileOptions's
// fastMathEnabled left at its default true, see ggml-metal-device.m) -- it
// matches ggml-metal's own pre-existing kernel_soft_max, and the masked path
// always `continue`s rather than ever evaluating exp(-INFINITY), so it does
// not depend on fast-math's Inf/NaN-may-not-occur assumption the way a naive
// exp(-inf) call would. Recorded here so a future toolchain change that
// breaks this is diagnosable rather than mysterious.

static inline float flash_attn_train_mask_val(
        device const half * mp,
        int32_t has_mask,
        int32_t mne0, int32_t mne1, int32_t mne2, int32_t mne3,
        int64_t h, int64_t b, int64_t i, int64_t j) {
    if (!has_mask) {
        return 0.0f;
    }
    // modulo, not divide -- ggml_soft_max_ext's own broadcast rule, carried
    // over unchanged from ggml_fa_train_mask_val in ggml-cpu/ops.cpp.
    const int64_t idx = j + mne0*(i + mne1*((h % mne2) + mne2*(b % mne3)));
    return (float) mp[idx];
}

// B6b: with a causal hint the mask is exactly "visible iff j <= prefix + i" (0 or -inf), so the
// causal kernel variants compute it instead of loading it from device memory. CAUSAL is a template
// flag (separate pipelines), so the generic variants carry no extra runtime branch.
template <bool CAUSAL>
static inline float flash_attn_train_mask_val_t(
        int32_t causal_prefix,
        device const half * mp,
        int32_t has_mask,
        int32_t mne0, int32_t mne1, int32_t mne2, int32_t mne3,
        int64_t h, int64_t b, int64_t i, int64_t j) {
    if (CAUSAL) {
        return j <= (int64_t) causal_prefix + i ? 0.0f : -INFINITY;
    }
    return flash_attn_train_mask_val(mp, has_mask, mne0, mne1, mne2, mne3, h, b, i, j);
}

// HOT-Step: FLASH_ATTN_TRAIN (kernel #7), register-accumulator redesign
// (2026-09-22 Opus review, "P1 performance redesign" -- see
// docs/plans/yue2-joint-training-metal-port.md).
//
// v1 was "one thread owns one whole row, two full O(S_kv*D) passes, no
// threadgroup memory" -- correct, but every thread re-reads the SAME K/V
// rows from device memory independently, and the output accumulator lived
// in device memory (dst itself) the whole time. This version ports the CUDA
// backend's own plain-F32 warp kernels (fa_train_fwd_f32 /
// fa_train_bwd_dq_f32 / fa_train_bwd_dkdv_f32 / fa_train_bwd_delta_f32 in
// ggml-cuda/fattn-train.cu, each template<int D>) to Metal's simdgroup
// primitives one-for-one:
//
//   * one SIMDGROUP (N_SIMDWIDTH = 32 lanes) owns one "own" row -- a query
//     row in the forward and in dQ below, a KV row in dK/dV -- exactly like
//     one CUDA warp owns one row.
//   * each lane holds D/32 accumulator floats in REGISTERS (thread-local
//     arrays), combined via simd_sum/simd_max/simd_shuffle instead of a
//     device-memory round trip per key -- CUDA's warp_reduce_*/__shfl_sync.
//   * the OTHER side (K/V in the forward and dQ, Q/dO in dK/dV) is staged
//     into THREADGROUP memory HOT_STEP_FA_TRAIN_BK rows at a time and
//     shared by every simdgroup in the threadgroup, so it is read from
//     device memory ONCE per tile per threadgroup rather than once per
//     thread.
//   * di = sum_d dO[d]*O[d] (CUDA's "delta") is precomputed once per row by
//     a dedicated kernel_flash_attn_train_delta_f32 pass and read back by
//     both backward kernels, rather than recomputed per (h,i) on every KV
//     row that reads it the way v1's dK/dV pass did (that kernel's own
//     comment called this out as "a follow-up can cache it" -- this is
//     that follow-up).
//   * D is a compile-time template parameter, D in {64, 128} -- matching
//     the CUDA port's own scope (fattn-train.cu's dispatch switches on
//     exactly these two and GGML_ABORTs otherwise; this project's own
//     head_dim is 128, yue2-aitk-graph.h) -- which is what makes the
//     accumulator arrays and the threadgroup tiles fixed-size, satisfying
//     MSL's ban on runtime-sized automatic/threadgroup arrays, the same
//     constraint v1 sidestepped by using device memory instead of
//     registers at all. ggml_metal_library_get_pipeline_flash_attn_train{,_back_dq,_back_dkdv,_delta}
//     picks the D=64 or D=128 pipeline by q->ne[0]; ggml_metal_device_supports_op
//     falls through to another backend for any other head dim.
//
// HOT_STEP_FA_TRAIN_BK is 16, not CUDA's 32, to fit comfortably inside a
// 32 KB threadgroup-memory budget at D = 128: the K/V (or Q/dO) tile alone
// is 2*16*(128+1)*4 = 16512 B at BK=16, versus 2*32*129*4 = 33024 B at
// BK=32 -- already over budget before the per-simdgroup "own row" tile is
// even added. Total threadgroup memory at D=128 across all three tiled
// kernels tops out around 20.8 KB (dK/dV, the largest) -- see the by-hand
// accounting in this patch's own commit/PR notes -- comfortably inside the
// M1 Max's declared per-threadgroup budget with margin for a future tuning
// pass, which is why there is no additional runtime smem-vs-device-limit
// assert here (D is fixed to {64,128} by the pipeline picker above, so this
// is a closed, hand-verified case, not an open-ended one).
#define HOT_STEP_FA_TRAIN_NSG 12   // "own" rows (simdgroups) per threadgroup -- was 4;
                                    // raised to cut how often a query-block re-sweeps
                                    // the full K/V sequence from device memory (S/NSG
                                    // sweeps/head; at NSG=4, S~15.5k that's ~3874 full
                                    // re-reads of a ~16MB-per-head K/V table -- the
                                    // dominant cost per Instruments' GPU counters:
                                    // Buffer Read Limiter ~94%, ALU Utilization ~10%,
                                    // measured 2026-09-26, Axel.
                                    // Capped at 12, not a rounder 16: the two backward
                                    // kernels (dQ, dK/dV) each carry a fixed ~16.5KB of
                                    // other threadgroup buffers (two BK=16-row tiles)
                                    // plus ~1036 B/NSG of their own -- 16 pushes both
                                    // past the 32KB static threadgroup-memory limit,
                                    // which would fail to compile/link, not just run slow.
#define HOT_STEP_FA_TRAIN_BK  16   // far-side rows staged per tile
// B4-E3 (2026-09-30): the dQ and dK/dV kernels get their OWN simdgroup count.
// Smem at D=128: 16512 B fixed tile + 1032 B per simdgroup + ~60 B -> BNSG 15
// = ~32.0 KB (dQ) / ~32.2 KB (dK/dV), BNSG 16 does not fit 32 KB. Keep in sync
// with ggml-metal-ops.cpp. Forward and delta stay on HOT_STEP_FA_TRAIN_NSG.
#define HOT_STEP_FA_TRAIN_BNSG 15

// Forward. Grid (ceil(S/HOT_STEP_FA_TRAIN_NSG), Nh, Bn); one simdgroup per
// query row (i,h,b), NV = D/32 accumulator floats per lane, K/V tiled
// through threadgroup memory HOT_STEP_FA_TRAIN_BK rows at a time and shared
// by every simdgroup in the threadgroup.
template <int D>
kernel void kernel_flash_attn_train_f32(
        constant ggml_metal_kargs_flash_attn_train & args,
        device const char * q,    // [D, S,    Nh,  Bn] F32
        device const char * k,    // [D, S_kv, Nkv, Bn] F32
        device const char * v,    // [D, S_kv, Nkv, Bn] F32 (NOT transposed)
        device const char * mask, // [S_kv, mne1, mne2, mne3] F16, or q's own
                                   // buffer (unread) when args.has_mask == 0
        device       char * dst,  // packed O|LSE, see ggml-metal-impl.h
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NV = D/N_SIMDWIDTH;  // accumulator floats per lane
    constexpr short LD = D + 1;          // padded threadgroup tile row stride

    threadgroup float Ksh[HOT_STEP_FA_TRAIN_BK*LD];
    threadgroup float Vsh[HOT_STEP_FA_TRAIN_BK*LD];
    threadgroup float Qsh[HOT_STEP_FA_TRAIN_NSG*LD];
    threadgroup int   live_sh[HOT_STEP_FA_TRAIN_NSG];

    const short  lane = tiisg;
    const short  w    = sgitg;
    const ushort tid  = w*N_SIMDWIDTH + lane;

    // Lane-pair split: HOT_STEP_FA_TRAIN_BK (16, the K/V/Q tile's row count)
    // is exactly half of N_SIMDWIDTH (32), so pairing adjacent lanes lets
    // both halves of each simdgroup do useful work -- each pair of lanes
    // (2*trow, 2*trow+1) cooperates on tile row `trow`, splitting its
    // D-length dot product across `thalf` and recombining with
    // simd_shuffle(x, lane ^ 1). See the plan doc's "lane-pair-split"
    // section for the full rationale.
    static_assert(HOT_STEP_FA_TRAIN_BK * 2 == N_SIMDWIDTH, "lane-pair split assumes BK == N_SIMDWIDTH/2");
    const short  trow  = lane >> 1;
    const short  thalf = lane &  1;

    const int64_t Nh = args.Nh;
    const int64_t S  = args.S;

    const int64_t i  = tgpig.x*HOT_STEP_FA_TRAIN_NSG + w;
    const int64_t h  = tgpig.y;
    const int64_t b  = tgpig.z;
    const int64_t hk = h/args.G;

    const bool active = i < S;

    if (active) {
        device const float * qrow = (device const float *) (q + i*args.nb01 + h*args.nb02 + b*args.nb03);
        for (short c = 0; c < NV; ++c) {
            Qsh[w*LD + lane + N_SIMDWIDTH*c] = qrow[lane + N_SIMDWIDTH*c];
        }
    }
    // Qsh's row is only read back by the simdgroup that wrote it.
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float acc[NV];
    for (short c = 0; c < NV; ++c) {
        acc[c] = 0.0f;
    }
    float mrun = -INFINITY; // running row max
    float lrun = 0.0f;      // running sum of exp

    device const half * mp = (device const half *) mask;

    for (int64_t j0 = 0; j0 < args.S_kv; j0 += HOT_STEP_FA_TRAIN_BK) {
        // (A) previous tile's compute is done: safe to overwrite K/V and flags
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int64_t j   = j0 + trow;
        // Both lanes of a pair (2*trow, 2*trow+1) address the same tile
        // row `trow`, which is inherently < HOT_STEP_FA_TRAIN_BK (see the
        // static_assert above) -- no separate bound needed here.
        const bool    jok = active && (j < args.S_kv);

        float mv = 0.0f;
        if (jok) {
            mv = flash_attn_train_mask_val(mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i, j);
        }
        const bool live = jok && (mv != -INFINITY);

        const bool anyw = simd_any(live);
        if (lane == 0) {
            live_sh[w] = anyw ? 1 : 0;
        }

        // (B) tile liveness known to the whole threadgroup
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool block_live = false;
        for (short t = 0; t < HOT_STEP_FA_TRAIN_NSG; ++t) {
            block_live = block_live || (live_sh[t] != 0);
        }
        if (!block_live) {
            // Whole tile dead (masking, or the tail of S_kv). Every thread
            // in the threadgroup takes this branch uniformly, so the next
            // loop iteration's barrier (A) stays uniform too.
            continue;
        }

        const int64_t nk = (args.S_kv - j0 < HOT_STEP_FA_TRAIN_BK) ? (args.S_kv - j0) : (int64_t) HOT_STEP_FA_TRAIN_BK;
        // Vectorized: one float4 load covers 4 consecutive feature-dim
        // elements (D is a multiple of 4), 4x fewer load-pipeline
        // instructions for the same bytes. packed_float4 (not float4) for
        // the threadgroup store: LD=D+1 staggers each row's start to dodge
        // bank conflicts, so jj*LD isn't always 16B-aligned -- packed_float4
        // has no alignment requirement, float4 there would be UB.
        constexpr short D4 = D/4;
        for (short idx = tid; idx < HOT_STEP_FA_TRAIN_BK*D4; idx += HOT_STEP_FA_TRAIN_NSG*N_SIMDWIDTH) {
            const short jj  = idx/D4;
            const short dd4 = idx%D4;
            float4 kval = float4(0.0f);
            float4 vval = float4(0.0f);
            if (jj < nk) {
                device const float4 * krow = (device const float4 *) (k + (j0 + jj)*args.nb11 + hk*args.nb12 + b*args.nb13);
                device const float4 * vrow = (device const float4 *) (v + (j0 + jj)*args.nb21 + hk*args.nb22 + b*args.nb23);
                kval = krow[dd4];
                vval = vrow[dd4];
            }
            *(threadgroup packed_float4 *) (Ksh + jj*LD + dd4*4) = packed_float4(kval);
            *(threadgroup packed_float4 *) (Vsh + jj*LD + dd4*4) = packed_float4(vval);
        }

        // (C) K/V tile staged
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s = -INFINITY;
        if (live) {
            threadgroup const float * qs = Qsh + w*LD;
            threadgroup const float * ks = Ksh + trow*LD;
            // Lane-pair split: each lane sums half of the D-length dot
            // product (stride 2, offset by thalf) and recombines with its
            // partner via simd_shuffle(., lane ^ 1) -- both lanes of the
            // pair end up with the identical, full-precision dot product
            // (IEEE-754 addition is commutative, so this matches bit for
            // bit regardless of which lane's half is added first).
            float dot_half = 0.0f;
            for (short d = thalf; d < D; d += 2) {
                dot_half += qs[d]*ks[d];
            }
            const float dot = dot_half + simd_shuffle(dot_half, lane ^ 1);
            s = args.scale*dot + mv;
        }

        // warp-uniform after the reduction; mrun is warp-uniform by induction
        const float tilemax = simd_max(s);
        const float m_new   = max(mrun, tilemax);
        if (m_new == -INFINITY) {
            continue;   // still nothing seen on this row
        }

        // exp(-INFINITY - finite) is 0 in IEEE-754; spelled out rather than
        // evaluated, matching this file's own fast-math note above.
        const float corr = (mrun == -INFINITY) ? 0.0f : exp(mrun - m_new);
        const float p    = (s    == -INFINITY) ? 0.0f : exp(s    - m_new);

        // p is duplicated across each lane pair (both lanes computed the
        // same row); zero thalf==1's copy so simd_sum below counts every
        // tile row exactly once.
        const float p_reduce = (thalf == 0) ? p : 0.0f;

        lrun = lrun*corr + simd_sum(p_reduce);
        for (short c = 0; c < NV; ++c) {
            acc[c] *= corr;
        }

        // Fixed source lane (2*r, the thalf==0 half of row r's pair) per
        // row, so the accumulation order is fixed too.
        for (short r = 0; r < HOT_STEP_FA_TRAIN_BK; ++r) {
            const float pr = simd_shuffle(p, 2*r);
            if (pr == 0.0f) {
                continue;   // masked or vanished: contributes exactly nothing
            }
            threadgroup const float * vs = Vsh + r*LD;
            for (short c = 0; c < NV; ++c) {
                acc[c] += pr*vs[lane + N_SIMDWIDTH*c];
            }
        }

        mrun = m_new;
    }

    if (!active) {
        return;
    }

    const int64_t  row  = h + Nh*(i + S*b);
    device float * orow = (device float *) (dst + row*D*sizeof(float));
    device float * lse  = (device float *) (dst + args.offs_lse);

    if (lrun > 0.0f) {
        for (short c = 0; c < NV; ++c) {
            orow[lane + N_SIMDWIDTH*c] = acc[c]/lrun;
        }
        if (lane == 0) {
            lse[row] = mrun + log(lrun);
        }
    } else {
        // spec 4.4: fully-masked row -- defined, finite, NOT NaN
        for (short c = 0; c < NV; ++c) {
            orow[lane + N_SIMDWIDTH*c] = 0.0f;
        }
        if (lane == 0) {
            lse[row] = 0.0f;
        }
    }
}

typedef decltype(kernel_flash_attn_train_f32<64>) kernel_flash_attn_train_f32_t;
template [[host_name("kernel_flash_attn_train_f32_d64" )]] kernel kernel_flash_attn_train_f32_t kernel_flash_attn_train_f32<64>;
template [[host_name("kernel_flash_attn_train_f32_d128")]] kernel kernel_flash_attn_train_f32_t kernel_flash_attn_train_f32<128>;

// ──────────────────────────────────────────────────────────────────────────
// EXPERIMENTAL -- simdgroup_matrix rewrite of FLASH_ATTN_TRAIN's forward
// kernel. Opt-in via GGML_METAL_FA_TRAIN_MM=1 (default OFF). NOT YET
// NUMERICALLY VALIDATED against kernel_flash_attn_train_f32 -- see
// docs/perf-notes/fa-train-simdgroup-matrix-plan.md for the plan and
// docs/perf-notes/fa-train-mm-validation.md (added alongside this kernel)
// for the required validation steps BEFORE this is ever used for a real
// training run. Forward-only: the backward kernels are untouched and only
// need a correct O|LSE region to read, which this kernel produces in the
// exact same packed layout as the scalar kernel.
//
// Design (see the plan doc for the full rationale): 8 query rows (NQ) x 8
// kv rows (NC) per simdgroup_matrix tile -- the GPU's native 8x8 matrix
// unit size, following this file's own kernel_convrot8_mm_f32 for the
// simdgroup_float8x8 / simdgroup_load / simdgroup_multiply_accumulate /
// simdgroup_store API. HOT_STEP_FA_TRAIN_MM_NSG simdgroups per threadgroup,
// each owning its own NQ query rows and computing independently against a
// K/V tile shared (via threadgroup memory) by the whole threadgroup --
// same K/V-sharing structure as kernel_flash_attn_train_f32, just NC=8
// instead of BK=16 (NC is no longer pinned by a lane-pair split, since
// there is no lane-pair split here).
//
// The online-softmax running-max/running-sum bookkeeping (mrun/lrun/corr)
// is done in SCALAR per-row form, via a small per-simdgroup 8x8
// threadgroup scratch buffer that the matrix tile is stored into and
// reloaded from -- this keeps the numerically-sensitive part of the
// algorithm textually close to kernel_flash_attn_train_f32 (same
// mrun/corr/p formulas) and confines the new, unverified risk to the two
// GEMMs (S = Q*K^T and O += P*V) and their load/store plumbing. This is a
// deliberate correctness-over-performance tradeoff for a first draft: the
// O-accumulator correction (multiplying every accumulated chunk by corr[r]
// on every KV tile) round-trips through threadgroup memory instead of a
// single per-lane multiply the way the scalar kernel does it, so this
// kernel's actual throughput relative to the scalar one is UNKNOWN until
// measured -- do not assume it is faster without profiling it.
//
// Known risk areas to check first if numerical validation fails:
//   - simdgroup_load's `transpose` argument on the K load (must yield K^T
//     so that Q(8xD) * K^T(Dx8) tiles correctly -- kernel_convrot8_mm_f32's
//     `mw` load is the reference for the transpose-argument position).
//   - Row/col orientation of simdgroup_store's output into Ssh/Csh
//     (row-major, stride = the tile width passed to store/load).
//   - Boundary handling for i0+r >= S (padded query rows) and
//     j0+c >= S_kv (padded/tail kv rows) -- both are masked to -INFINITY
//     rather than skipped, unlike the scalar kernel's `active`/`nk` early
//     exits, so double-check padded rows never got written to the O/LSE
//     buffers (guarded below) and never corrupt a live row's reduction
//     (they can't: matrix rows are independent, and only the mask value
//     -- never the padded Q/K/V payload -- decides each row's own p).
//   - v2 step 1 (2026-09-27): masked-tile-skip has been reintroduced (see
//     live_sh/block_live below, ported from kernel_flash_attn_train_f32).
//     Whole-tile-masked KV iterations now skip K/V staging and both GEMMs
//     entirely instead of always staging and multiplying with p=0.
//   - v2 step 2 (2026-09-27): the O-accumulator correction's store/scale/
//     load round-trip (described above) is now conditional -- skipped for
//     the whole threadgroup on any tile where every row's running max was
//     already stable (corr_local == 1.0f everywhere, so acc *= corr would
//     be a no-op). Still round-trips through threadgroup memory on tiles
//     where it IS needed; a true per-lane multiply like the scalar
//     kernel's is what step "cheapen the O-accumulator correction" in the
//     v2 plan calls the higher-ceiling but MSL-API-unconfirmed follow-up.
#define HOT_STEP_FA_TRAIN_MM_NQ  8   // query rows per simdgroup tile
#define HOT_STEP_FA_TRAIN_MM_NC  8   // kv rows per tile (native 8x8 unit)
#define HOT_STEP_FA_TRAIN_MM_NSG 2   // simdgroups per threadgroup (memory-budget-limited, see below)

template <int D>
kernel void kernel_flash_attn_train_mm_f32(
        constant ggml_metal_kargs_flash_attn_train & args,
        device const char * q,    // [D, S,    Nh,  Bn] F32
        device const char * k,    // [D, S_kv, Nkv, Bn] F32
        device const char * v,    // [D, S_kv, Nkv, Bn] F32 (NOT transposed)
        device const char * mask, // [S_kv, mne1, mne2, mne3] F16, or q's own
                                   // buffer (unread) when args.has_mask == 0
        device       char * dst,  // packed O|LSE, see ggml-metal-impl.h
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NQ  = HOT_STEP_FA_TRAIN_MM_NQ;
    constexpr short NC  = HOT_STEP_FA_TRAIN_MM_NC;
    constexpr short NSG = HOT_STEP_FA_TRAIN_MM_NSG;
    constexpr short NV  = D/8;   // number of 8-wide column chunks across D
    constexpr short LD  = D + 1; // padded row stride (same anti-bank-conflict trick as the scalar kernel)

    threadgroup float * Ksh = (threadgroup float *) shmem;                // [NC][LD]
    threadgroup float * Vsh = Ksh + NC*LD;                                // [NC][LD]
    threadgroup float * Qsh = Vsh + NC*LD;                                // [NSG*NQ][LD]
    threadgroup float * Ssh = Qsh + NSG*NQ*LD;                            // [NSG][NQ*NC] per-simdgroup score/prob scratch
    threadgroup float * Csh = Ssh + NSG*NQ*NC;                            // [NSG][NV*NQ*8] per-simdgroup acc-correction scratch
    threadgroup int   * live_sh = (threadgroup int *) (Csh + NSG*NV*NQ*8); // [NSG] masked-tile-skip liveness, one flag per simdgroup

    const short lane = tiisg;
    const short w    = sgitg;
    const ushort tid = w*N_SIMDWIDTH + lane;

    const int64_t Nh = args.Nh;
    const int64_t S  = args.S;

    const int64_t i0 = tgpig.x*(NSG*NQ) + w*NQ; // first query row owned by this simdgroup
    const int64_t h  = tgpig.y;
    const int64_t b  = tgpig.z;
    const int64_t hk = h/args.G;

    threadgroup float * qsh_w = Qsh + w*NQ*LD;
    threadgroup float * ssh_w = Ssh + w*NQ*NC;
    threadgroup float * csh_w = Csh + w*NV*NQ*8;

    // Stage this simdgroup's own up-to-NQ query rows (not shared with other
    // simdgroups, unlike K/V below) -- padded rows (i >= S) get zeros so
    // the matrix loads never read uninitialized threadgroup memory, but
    // their score/output is discarded later via explicit i<S guards.
    for (short r = 0; r < NQ; ++r) {
        const int64_t i = i0 + r;
        const bool row_ok = i < S;
        device const float4 * qrow = row_ok
            ? (device const float4 *) (q + i*args.nb01 + h*args.nb02 + b*args.nb03)
            : nullptr;
        for (short c4 = lane; c4 < D/4; c4 += N_SIMDWIDTH) {
            const float4 qv = row_ok ? qrow[c4] : float4(0.0f);
            *(threadgroup packed_float4 *) (qsh_w + r*LD + c4*4) = packed_float4(qv);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[NV];
    for (short c = 0; c < NV; ++c) {
        acc[c] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    float mrun[NQ];
    float lrun[NQ];
    for (short r = 0; r < NQ; ++r) {
        mrun[r] = -INFINITY;
        lrun[r] = 0.0f;
    }

    device const half * mp = (device const half *) mask;

    for (int64_t j0 = 0; j0 < args.S_kv; j0 += NC) {
        // (A) previous tile's compute is done: safe to overwrite K/V.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Masked-tile-skip (v2 step 1, ported from kernel_flash_attn_train_f32's
        // live_sh/block_live pattern). Mask lookups only depend on (h, b, i, j),
        // not on K/V data, so liveness can be checked BEFORE staging K/V --
        // skipping the whole tile (K/V load + both GEMMs) whenever no query
        // row owned by ANY simdgroup in this threadgroup has a live entry
        // against this tile's NC columns. At ~15k-token causal sequences
        // roughly half of all tiles are fully masked.
        bool row_live = false;
        if (lane < NQ) {
            const int64_t i = i0 + lane;
            if (i < S) {
                for (short c = 0; c < NC; ++c) {
                    const int64_t j = j0 + c;
                    if (j < args.S_kv) {
                        const float mv = flash_attn_train_mask_val(mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i, j);
                        if (mv != -INFINITY) {
                            row_live = true;
                            break;
                        }
                    }
                }
            }
        }
        const bool anyw = simd_any(row_live);
        if (lane == 0) {
            live_sh[w] = anyw ? 1 : 0;
        }

        // (A2) tile liveness known to the whole threadgroup.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool block_live = false;
        for (short t = 0; t < NSG; ++t) {
            block_live = block_live || (live_sh[t] != 0);
        }
        if (!block_live) {
            // Whole tile dead for every query row this threadgroup owns.
            // live_sh is fully populated post-barrier, so this branch is
            // uniform across the whole threadgroup -- the next iteration's
            // barrier (A) stays uniform too, same reasoning as the scalar
            // kernel's block_live check.
            continue;
        }

        const int64_t nk = (args.S_kv - j0 < NC) ? (args.S_kv - j0) : (int64_t) NC;

        // Stage the K/V tile, shared by every simdgroup in the threadgroup
        // (same vectorized float4/packed_float4 pattern as the scalar
        // kernel's own K/V staging loop).
        constexpr short D4 = D/4;
        for (short idx = tid; idx < NC*D4; idx += NSG*N_SIMDWIDTH) {
            const short jj  = idx/D4;
            const short dd4 = idx%D4;
            float4 kval = float4(0.0f);
            float4 vval = float4(0.0f);
            if (jj < nk) {
                device const float4 * krow = (device const float4 *) (k + (j0 + jj)*args.nb11 + hk*args.nb12 + b*args.nb13);
                device const float4 * vrow = (device const float4 *) (v + (j0 + jj)*args.nb21 + hk*args.nb22 + b*args.nb23);
                kval = krow[dd4];
                vval = vrow[dd4];
            }
            *(threadgroup packed_float4 *) (Ksh + jj*LD + dd4*4) = packed_float4(kval);
            *(threadgroup packed_float4 *) (Vsh + jj*LD + dd4*4) = packed_float4(vval);
        }

        // (B) K/V tile staged.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = scale * Q*K^T, accumulated over D in 8-wide chunks.
        simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (short c = 0; c < NV; ++c) {
            simdgroup_float8x8 mq, mkt;
            simdgroup_load(mq,  qsh_w + c*8, LD);
            // transposed load: K's [row][d-chunk] tile -> (d-chunk x row) = K^T
            simdgroup_load(mkt, Ksh   + c*8, LD, 0, true);
            simdgroup_multiply_accumulate(sacc, mq, mkt, sacc);
        }
        simdgroup_store(sacc, ssh_w, NC);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Scalar softmax step, one lane per query row (lanes NQ..31 idle
        // here -- correctness first; see the plan doc for later
        // parallelizing this across all 32 lanes).
        float corr_local = 1.0f; // only meaningful for lane < NQ
        if (lane < NQ) {
            const short r  = lane;
            const int64_t i = i0 + r;
            float tilemax = -INFINITY;
            float srow[NC];
            for (short c = 0; c < NC; ++c) {
                const int64_t j = j0 + c;
                float s = -INFINITY;
                if (i < S && j < args.S_kv) {
                    const float mv = flash_attn_train_mask_val(mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i, j);
                    if (mv != -INFINITY) {
                        s = args.scale*ssh_w[r*NC + c] + mv;
                    }
                }
                srow[c] = s;
                tilemax = max(tilemax, s);
            }
            const float m_new = max(mrun[r], tilemax);
            if (m_new != -INFINITY) {
                const float corr = (mrun[r] == -INFINITY) ? 0.0f : exp(mrun[r] - m_new);
                corr_local = corr;
                float lsum = 0.0f;
                for (short c = 0; c < NC; ++c) {
                    const float p = (srow[c] == -INFINITY) ? 0.0f : exp(srow[c] - m_new);
                    ssh_w[r*NC + c] = p; // reuse the score scratch as the P matrix
                    lsum += p;
                }
                lrun[r] = lrun[r]*corr + lsum;
                mrun[r] = m_new;
            } else {
                corr_local = 1.0f; // row saw nothing yet again this tile
                for (short c = 0; c < NC; ++c) {
                    ssh_w[r*NC + c] = 0.0f;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup); // P matrix in ssh_w visible to the whole simdgroup

        // (v2 step 2) O-accumulator correction skip: once every row's
        // running max has stabilized, corr_local == 1.0f for the whole
        // simdgroup and the store/scale/load round-trip below is a no-op
        // (acc *= 1). Reduced to a per-tile, WHOLE-THREADGROUP decision --
        // not per-simdgroup -- purely so every simdgroup takes the same
        // branch below in lockstep (the simdgroup_barrier calls inside it
        // only need their own 32 lanes to agree, but a threadgroup-uniform
        // branch keeps that trivially true without per-simdgroup reasoning
        // about it). Reusing live_sh here is safe: it was already consumed
        // by the tile-skip check above, before K/V staging, and isn't read
        // again until next iteration's barrier (A).
        const bool corr_needed_sg = simd_any(corr_local != 1.0f);
        if (lane == 0) {
            live_sh[w] = corr_needed_sg ? 1 : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool corr_needed = false;
        for (short t = 0; t < NSG; ++t) {
            corr_needed = corr_needed || (live_sh[t] != 0);
        }

        // Rescale every accumulated chunk by this tile's per-row corr
        // BEFORE adding this tile's P*V contribution (same order as the
        // scalar kernel: acc *= corr, then acc += P*V) -- skipped
        // entirely when no row in this threadgroup needs a rescale.
        if (corr_needed) {
            // corr_needed is identical across every lane in the threadgroup
            // (reduced via live_sh + a threadgroup_barrier above), so this
            // branch is taken uniformly -- the simdgroup-scoped barriers
            // below (same as the original, unconditional version) stay
            // valid: csh_w is this simdgroup's own private scratch, never
            // touched by another simdgroup, so only its own 32 lanes need
            // to agree, not the whole threadgroup.
            for (short c = 0; c < NV; ++c) {
                simdgroup_store(acc[c], csh_w + c*NQ*8, 8);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (short idx = lane; idx < NV*NQ*8; idx += N_SIMDWIDTH) {
                const short row = (idx/8) % NQ;
                // corr_local is only valid on lanes < NQ; broadcast each row's
                // corr from its owning lane so every lane can apply it here.
                const float corr_row = simd_shuffle(corr_local, row);
                csh_w[idx] *= corr_row;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (short c = 0; c < NV; ++c) {
                simdgroup_load(acc[c], csh_w + c*NQ*8, 8);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        // acc += P * V, tiled over D in 8-wide chunks.
        simdgroup_float8x8 pmat;
        simdgroup_load(pmat, ssh_w, NC);
        for (short c = 0; c < NV; ++c) {
            simdgroup_float8x8 mv8;
            simdgroup_load(mv8, Vsh + c*8, LD);
            simdgroup_multiply_accumulate(acc[c], pmat, mv8, acc[c]);
        }
    }

    // Write O and LSE, one 8x8 chunk of the accumulator at a time.
    for (short c = 0; c < NV; ++c) {
        simdgroup_store(acc[c], ssh_w, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < NQ) {
            const short  r = lane;
            const int64_t i = i0 + r;
            if (i < S) {
                const int64_t  row  = h + Nh*(i + S*b);
                device float * orow = (device float *) (dst + row*D*sizeof(float));
                if (lrun[r] > 0.0f) {
                    for (short cc = 0; cc < 8; ++cc) {
                        orow[c*8 + cc] = ssh_w[r*8 + cc]/lrun[r];
                    }
                } else {
                    // spec 4.4: fully-masked row -- defined, finite, NOT NaN
                    for (short cc = 0; cc < 8; ++cc) {
                        orow[c*8 + cc] = 0.0f;
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < NQ) {
        const short  r = lane;
        const int64_t i = i0 + r;
        if (i < S) {
            const int64_t  row = h + Nh*(i + S*b);
            device float * lse = (device float *) (dst + args.offs_lse);
            lse[row] = (lrun[r] > 0.0f) ? (mrun[r] + log(lrun[r])) : 0.0f;
        }
    }
}

typedef decltype(kernel_flash_attn_train_mm_f32<64>) kernel_flash_attn_train_mm_f32_t;
template [[host_name("kernel_flash_attn_train_mm_f32_d64" )]] kernel kernel_flash_attn_train_mm_f32_t kernel_flash_attn_train_mm_f32<64>;
template [[host_name("kernel_flash_attn_train_mm_f32_d128")]] kernel kernel_flash_attn_train_mm_f32_t kernel_flash_attn_train_mm_f32<128>;

// ──────────────────────────────────────────────────────────────────────────
// EXPERIMENTAL (B3, 2026-09-30) -- simdgroup_matrix forward, v3: TWO-PASS.
// Opt-in via GGML_METAL_FA_TRAIN_MM3=1 (default OFF). Tolerance-gated, not
// bit-identical to kernel_flash_attn_train_f32 (different summation order).
//
// Why v3 and not another patch of kernel_flash_attn_train_mm_f32 (v1/v2 were
// correct but 1.37-1.5x SLOWER than the scalar kernel): v1's costs were (a)
// the O-accumulator correction (acc *= corr) round-tripping through
// threadgroup memory once per KV tile, (b) an 8-key tile = 4 threadgroup
// barriers per 16 MMAs, (c) softmax and mask handling on 8 of 32 lanes,
// (d) 2 simdgroups per threadgroup. v3 removes (a) by construction:
//   pass 1: S = scale*Q*K^T tile by tile, online (row max m, row sum l) only --
//           scalar per-row state, NO output accumulator exists yet;
//   pass 2: S again, P = exp(S - LSE) with the FINAL LSE = m + log(l), so
//           O += P*V needs no rescaling at all (this is exactly how the two
//           backward kernels already recompute P from the LSE).
// Cost: 3 matmuls instead of 2 (1.5x FLOPs), recovered by (b), (c), (d):
// 16-key tiles, all 32 lanes share the softmax (8 rows x 4 lanes), NSG = 8
// simdgroups of 8 query rows each (64 rows per threadgroup share every K/V
// tile; the scalar kernel shares a tile across 12 rows).
//
// Output equals the scalar kernel's contract: O [D, Nh, S, Bn] and LSE in the
// packed layout; a fully masked row gives O = 0, LSE = 0 (spec 4.4).
// Q rows are staged once through threadgroup memory (aliasing the K/V tile
// buffer, padded rows = zeros so nothing reads out of bounds) into register
// fragments mq[D/8]; K/V tiles are staged per tile exactly like the scalar
// kernel (pass 1 stages K only). Fragment <-> element mapping is never
// assumed: every per-row step goes through a small per-simdgroup S scratch.
#define HOT_STEP_FA_TRAIN_MM3_NSG 8    // simdgroups per threadgroup, 8 query rows each; keep in sync with ggml-metal-ops.cpp
#define HOT_STEP_FA_TRAIN_MM3_NC  16   // KV rows per tile (2 column blocks of 8); keep in sync with ggml-metal-ops.cpp
#define HOT_STEP_FA_TRAIN_SPAD    1    // row padding of the S scratch (forward, dQ, dK, dV); keep in sync with ggml-metal-ops.cpp

template <int D, bool CAUSAL>
kernel void kernel_flash_attn_train_mm3_f32(
        constant ggml_metal_kargs_flash_attn_train & args,
        device const char * q,    // [D, S,    Nh,  Bn] F32
        device const char * k,    // [D, S_kv, Nkv, Bn] F32
        device const char * v,    // [D, S_kv, Nkv, Bn] F32 (NOT transposed)
        device const char * mask, // [S_kv, mne1, mne2, mne3] F16, or q's own
                                   // buffer (unread) when args.has_mask == 0
        device       char * dst,  // packed O|LSE, see ggml-metal-impl.h
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = HOT_STEP_FA_TRAIN_MM3_NSG;
    constexpr short NC  = HOT_STEP_FA_TRAIN_MM3_NC;
    constexpr short NCB = NC/8;      // column blocks per KV tile
    constexpr short NCP = NC + HOT_STEP_FA_TRAIN_SPAD;   // padded stride of the S/P scratch
    constexpr short NV  = D/8;       // 8-wide chunks across D
    constexpr short LD  = D + 1;     // padded K/V tile row stride
    constexpr short D4  = D/4;

    // smem: [Ksh NC*LD][Vsh NC*LD][Ssh NSG*8*NC][live NSG ints]
    threadgroup float * Ksh = (threadgroup float *) shmem;
    threadgroup float * Vsh = Ksh + NC*LD;
    threadgroup float * Ssh = Vsh + NC*LD;
    threadgroup int   * live_sh = (threadgroup int *) (Ssh + NSG*8*NCP);

    const short  lane = tiisg;
    const short  w    = sgitg;
    const ushort tid  = w*N_SIMDWIDTH + lane;

    const int64_t Nh = args.Nh;
    const int64_t S  = args.S;

    // B6: with a causal hint the heavy (late) query blocks are scheduled first, and the key loop
    // stops at the last key any row of this threadgroup can see (threadgroup-uniform).
    const int64_t nblk = (args.S + NSG*8 - 1)/(NSG*8);
    const int64_t blk  = CAUSAL ? nblk - 1 - (int64_t) tgpig.x : (int64_t) tgpig.x;
    const int64_t i0 = blk*(NSG*8) + w*8; // first query row of this simdgroup
    int64_t j_end = args.S_kv;
    if (CAUSAL) {
        const int64_t i_last = min((int64_t) args.S - 1, blk*(NSG*8) + NSG*8 - 1);
        j_end = min((int64_t) args.S_kv, (int64_t) args.causal_prefix + i_last + 1);
    }
    const int64_t h  = tgpig.y;
    const int64_t b  = tgpig.z;
    const int64_t hk = h/args.G;

    threadgroup float * ssh_w = Ssh + w*8*NCP;

    // per-lane softmax role: row r (8 rows), 4 columns cq..cq+3 of the 16-wide tile
    const short   r     = lane >> 2;
    const short   cq    = (lane & 3)*4;
    const int64_t i_row = i0 + r;

    // ---- stage Q (two halves through this simdgroup's private slice of the K/V buffer) ----
    simdgroup_float8x8 mq[NV];
    {
        threadgroup float * qs = Ksh + w*(8*(D/2)); // 8 rows x D/2 floats per simdgroup; NSG*8*(D/2) <= 2*NC*LD
        #pragma clang loop unroll(full)
        for (short hf = 0; hf < 2; ++hf) {
            for (short idx = lane; idx < 8*(D/8); idx += N_SIMDWIDTH) {
                const short rr = idx/(D/8);
                const short c4 = idx%(D/8);
                float4 qv = float4(0.0f);
                if (i0 + rr < S) {
                    device const float4 * qrow = (device const float4 *) (q + (i0 + rr)*args.nb01 + h*args.nb02 + b*args.nb03);
                    qv = qrow[hf*(D/8) + c4];
                }
                *(threadgroup packed_float4 *) (qs + rr*(D/2) + c4*4) = packed_float4(qv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (short c = 0; c < NV/2; ++c) {
                simdgroup_load(mq[hf*(NV/2) + c], qs + c*8, D/2);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    simdgroup_float8x8 acc[NV];
    #pragma clang loop unroll(full)
    for (short c = 0; c < NV; ++c) {
        acc[c] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    float mrun = -INFINITY; // this lane's row: running max  (identical on the row's 4 lanes)
    float lrun = 0.0f;      // running sum of exp
    float lse_r = 0.0f;
    bool  row_dead = true;

    device const half * mp = (device const half *) mask;

    for (short pass = 0; pass < 2; ++pass) {
        const bool second = (pass == 1);
        if (second) {
            row_dead = !(lrun > 0.0f);
            lse_r = row_dead ? 0.0f : (mrun + log(lrun));
        }

        for (int64_t j0 = 0; j0 < j_end; j0 += NC) {
            // (A) previous tile fully consumed: K/V buffer and flags may be overwritten
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (args.has_mask) {
                bool live = false;
                if (i_row < S) {
                    for (short c = 0; c < 4; ++c) {
                        const int64_t j = j0 + cq + c;
                        if (j < args.S_kv) {
                            const float mv = flash_attn_train_mask_val_t<CAUSAL>(args.causal_prefix, mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i_row, j);
                            if (mv != -INFINITY) {
                                live = true;
                            }
                        }
                    }
                }
                const bool anyw = simd_any(live);
                if (lane == 0) {
                    live_sh[w] = anyw ? 1 : 0;
                }
                // (B) liveness of every simdgroup visible
                threadgroup_barrier(mem_flags::mem_threadgroup);
                bool block_live = false;
                for (short t = 0; t < NSG; ++t) {
                    block_live = block_live || (live_sh[t] != 0);
                }
                if (!block_live) {
                    continue; // uniform across the threadgroup (live_sh read after the barrier)
                }
            }

            const int64_t nk = (args.S_kv - j0 < NC) ? (args.S_kv - j0) : (int64_t) NC;

            for (short idx = tid; idx < NC*D4; idx += NSG*N_SIMDWIDTH) {
                const short jj  = idx/D4;
                const short dd4 = idx%D4;
                float4 kval = float4(0.0f);
                float4 vval = float4(0.0f);
                if (jj < nk) {
                    device const float4 * krow = (device const float4 *) (k + (j0 + jj)*args.nb11 + hk*args.nb12 + b*args.nb13);
                    kval = krow[dd4];
                    if (second) {
                        device const float4 * vrow = (device const float4 *) (v + (j0 + jj)*args.nb21 + hk*args.nb22 + b*args.nb23);
                        vval = vrow[dd4];
                    }
                }
                *(threadgroup packed_float4 *) (Ksh + jj*LD + dd4*4) = packed_float4(kval);
                if (second) {
                    *(threadgroup packed_float4 *) (Vsh + jj*LD + dd4*4) = packed_float4(vval);
                }
            }

            // (C) tile staged
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // S = Q*K^T, 8 query rows x NC keys
            simdgroup_float8x8 sfrag[NCB];
            #pragma clang loop unroll(full)
            for (short cb = 0; cb < NCB; ++cb) {
                sfrag[cb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            #pragma clang loop unroll(full)
            for (short c = 0; c < NV; ++c) {
                #pragma clang loop unroll(full)
                for (short cb = 0; cb < NCB; ++cb) {
                    simdgroup_float8x8 mkt;
                    // transposed load of K rows [cb*8, cb*8+8) x d-chunk c  ->  (d-chunk x keys)
                    simdgroup_load(mkt, Ksh + cb*8*LD + c*8, LD, 0, true);
                    simdgroup_multiply_accumulate(sfrag[cb], mq[c], mkt, sfrag[cb]);
                }
            }
            #pragma clang loop unroll(full)
            for (short cb = 0; cb < NCB; ++cb) {
                simdgroup_store(sfrag[cb], ssh_w + cb*8, NCP);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // per-row softmax step: this lane owns row r, columns cq..cq+3
            float sv[4];
            float tmax = -INFINITY;
            for (short c = 0; c < 4; ++c) {
                const int64_t j = j0 + cq + c;
                float sc = -INFINITY;
                if (i_row < S && j < args.S_kv) {
                    const float mv = flash_attn_train_mask_val_t<CAUSAL>(args.causal_prefix, mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i_row, j);
                    if (mv != -INFINITY) {
                        sc = args.scale*ssh_w[r*NCP + cq + c] + mv;
                    }
                }
                sv[c] = sc;
                tmax = max(tmax, sc);
            }

            if (!second) {
                tmax = max(tmax, simd_shuffle(tmax, lane ^ 1));
                tmax = max(tmax, simd_shuffle(tmax, lane ^ 2));
                const float m_new = max(mrun, tmax);
                float ps = 0.0f;
                for (short c = 0; c < 4; ++c) {
                    ps += (sv[c] == -INFINITY || m_new == -INFINITY) ? 0.0f : exp(sv[c] - m_new);
                }
                ps += simd_shuffle(ps, lane ^ 1);
                ps += simd_shuffle(ps, lane ^ 2);
                if (m_new != -INFINITY) {
                    const float corr = (mrun == -INFINITY) ? 0.0f : exp(mrun - m_new);
                    lrun = lrun*corr + ps;
                    mrun = m_new;
                }
            } else {
                for (short c = 0; c < 4; ++c) {
                    const float pv = (row_dead || sv[c] == -INFINITY) ? 0.0f : exp(sv[c] - lse_r);
                    ssh_w[r*NCP + cq + c] = pv;   // P overwrites S in the scratch
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 pmat[NCB];
                #pragma clang loop unroll(full)
                for (short cb = 0; cb < NCB; ++cb) {
                    simdgroup_load(pmat[cb], ssh_w + cb*8, NCP);
                }
                #pragma clang loop unroll(full)
                for (short c = 0; c < NV; ++c) {
                    #pragma clang loop unroll(full)
                    for (short cb = 0; cb < NCB; ++cb) {
                        simdgroup_float8x8 mv8;
                        simdgroup_load(mv8, Vsh + cb*8*LD + c*8, LD);
                        simdgroup_multiply_accumulate(acc[c], pmat[cb], mv8, acc[c]);
                    }
                }
            }
        }
    }

    // ---- write O (one 8x8 chunk at a time) and LSE ----
    {
        // make sure the last P tile reads of every lane are done before ssh_w is reused
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (short c = 0; c < NV; ++c) {
            simdgroup_store(acc[c], ssh_w, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (i_row < S) {
                const int64_t  row  = h + Nh*(i_row + S*b);
                device float * orow = (device float *) (dst + row*D*sizeof(float));
                const short    cc   = (lane & 3)*2;
                orow[c*8 + cc    ] = row_dead ? 0.0f : ssh_w[r*8 + cc    ];
                orow[c*8 + cc + 1] = row_dead ? 0.0f : ssh_w[r*8 + cc + 1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (i_row < S && (lane & 3) == 0) {
            const int64_t  row = h + Nh*(i_row + S*b);
            device float * lse = (device float *) (dst + args.offs_lse);
            lse[row] = lse_r;   // 0 for a fully masked row
        }
    }
}

typedef decltype(kernel_flash_attn_train_mm3_f32<64, false>) kernel_flash_attn_train_mm3_f32_t;
template [[host_name("kernel_flash_attn_train_mm3_f32_d64" )]] kernel kernel_flash_attn_train_mm3_f32_t kernel_flash_attn_train_mm3_f32<64, false>;
template [[host_name("kernel_flash_attn_train_mm3_causal_f32_d64" )]] kernel kernel_flash_attn_train_mm3_f32_t kernel_flash_attn_train_mm3_f32<64, true>;
template [[host_name("kernel_flash_attn_train_mm3_f32_d128")]] kernel kernel_flash_attn_train_mm3_f32_t kernel_flash_attn_train_mm3_f32<128, false>;
template [[host_name("kernel_flash_attn_train_mm3_causal_f32_d128")]] kernel kernel_flash_attn_train_mm3_f32_t kernel_flash_attn_train_mm3_f32<128, true>;

// ──────────────────────────────────────────────────────────────────────────
// Weg 1 (2026-10-01) -- single-pass forward (online softmax),
// default ON since the Weg-1 flip; GGML_METAL_FA_TRAIN_MM3_1P=0 selects the two-pass kernel (needs MM3). Same structure as the two-pass kernel
// above, but only 2 matmuls per tile (QK^T, P*V) instead of 3. The O accumulator
// is corrected with a diagonal-matrix MMA (acc = diag(corr)*acc + P*V), like ggml's
// inference FA kernel, so there is no threadgroup round trip of the accumulator.
// Not bit-identical to the two-pass kernel (different summation/rounding order).
// NSG (simdgroups per threadgroup) and NC (KV rows per tile) are template parameters selected by the host
// (GGML_METAL_FA_TRAIN_FWD_NSG / _FWD_NC; defaults NSG 8, NC 8 for causal and 16 otherwise).

template <int D, bool CAUSAL, int NSG_ = 8, int NC_ = 16>
kernel void kernel_flash_attn_train_mm3_1p_f32(
        constant ggml_metal_kargs_flash_attn_train & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = NSG_;
    constexpr short NC  = NC_;
    constexpr short NCB = NC/8;
    constexpr short NCP = NC + HOT_STEP_FA_TRAIN_SPAD;
    constexpr short NV  = D/8;
    constexpr short LD  = D + 1;
    constexpr short D4  = D/4;

    // smem: [Ksh NC*LD][Vsh NC*LD][Ssh NSG*8*NCP][dsh NSG*64][live NSG ints]
    threadgroup float * Ksh = (threadgroup float *) shmem;
    threadgroup float * Vsh = Ksh + NC*LD;
    threadgroup float * Ssh = Vsh + NC*LD;
    threadgroup float * Dsh = Ssh + NSG*8*NCP;
    threadgroup int   * live_sh = (threadgroup int *) (Dsh + NSG*64);

    const short  lane = tiisg;
    const short  w    = sgitg;
    const ushort tid  = w*N_SIMDWIDTH + lane;

    const int64_t Nh = args.Nh;
    const int64_t S  = args.S;

    const int64_t nblk = (args.S + NSG*8 - 1)/(NSG*8);
    const int64_t blk  = CAUSAL ? nblk - 1 - (int64_t) tgpig.x : (int64_t) tgpig.x;
    const int64_t i0 = blk*(NSG*8) + w*8;
    int64_t j_end = args.S_kv;
    if (CAUSAL) {
        const int64_t i_last = min((int64_t) args.S - 1, blk*(NSG*8) + NSG*8 - 1);
        j_end = min((int64_t) args.S_kv, (int64_t) args.causal_prefix + i_last + 1);
    }
    const int64_t h  = tgpig.y;
    const int64_t b  = tgpig.z;
    const int64_t hk = h/args.G;

    threadgroup float * ssh_w = Ssh + w*8*NCP;
    threadgroup float * dsh_w = Dsh + w*64;

    const short   r     = lane >> 2;
    const short   CPL   = NC/4;           // columns per lane
    const short   cq    = (lane & 3)*CPL;
    const int64_t i_row = i0 + r;

    // zero the diagonal scratch (only the diagonal is rewritten per tile)
    dsh_w[lane]      = 0.0f;
    dsh_w[lane + 32] = 0.0f;

    simdgroup_float8x8 mq[NV];
    {
        constexpr short NPART = (NSG*8*(D/2) <= 2*NC*LD) ? 2 : ((NSG*8*(D/4) <= 2*NC*LD) ? 4 : 8);
        constexpr short CW    = D/NPART;
        threadgroup float * qs = Ksh + w*(8*CW);
        #pragma clang loop unroll(full)
        for (short hf = 0; hf < NPART; ++hf) {
            for (short idx = lane; idx < 8*(CW/4); idx += N_SIMDWIDTH) {
                const short rr = idx/(CW/4);
                const short c4 = idx%(CW/4);
                float4 qv = float4(0.0f);
                if (i0 + rr < S) {
                    device const float4 * qrow = (device const float4 *) (q + (i0 + rr)*args.nb01 + h*args.nb02 + b*args.nb03);
                    qv = qrow[hf*(CW/4) + c4];
                }
                *(threadgroup packed_float4 *) (qs + rr*CW + c4*4) = packed_float4(qv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (short c = 0; c < NV/NPART; ++c) {
                simdgroup_load(mq[hf*(NV/NPART) + c], qs + c*8, CW);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    simdgroup_float8x8 acc[NV];
    #pragma clang loop unroll(full)
    for (short c = 0; c < NV; ++c) {
        acc[c] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    simdgroup_float8x8 zero8 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    float mrun = -INFINITY;
    float lrun = 0.0f;

    device const half * mp = (device const half *) mask;

    for (int64_t j0 = 0; j0 < j_end; j0 += NC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (args.has_mask) {
            bool live = false;
            if (i_row < S) {
                for (short c = 0; c < CPL; ++c) {
                    const int64_t j = j0 + cq + c;
                    if (j < args.S_kv) {
                        const float mv = flash_attn_train_mask_val_t<CAUSAL>(args.causal_prefix, mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i_row, j);
                        if (mv != -INFINITY) {
                            live = true;
                        }
                    }
                }
            }
            const bool anyw = simd_any(live);
            if (lane == 0) {
                live_sh[w] = anyw ? 1 : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool block_live = false;
            for (short t = 0; t < NSG; ++t) {
                block_live = block_live || (live_sh[t] != 0);
            }
            if (!block_live) {
                continue;
            }
        }

        const int64_t nk = (args.S_kv - j0 < NC) ? (args.S_kv - j0) : (int64_t) NC;

        for (short idx = tid; idx < NC*D4; idx += NSG*N_SIMDWIDTH) {
            const short jj  = idx/D4;
            const short dd4 = idx%D4;
            float4 kval = float4(0.0f);
            float4 vval = float4(0.0f);
            if (jj < nk) {
                device const float4 * krow = (device const float4 *) (k + (j0 + jj)*args.nb11 + hk*args.nb12 + b*args.nb13);
                device const float4 * vrow = (device const float4 *) (v + (j0 + jj)*args.nb21 + hk*args.nb22 + b*args.nb23);
                kval = krow[dd4];
                vval = vrow[dd4];
            }
            *(threadgroup packed_float4 *) (Ksh + jj*LD + dd4*4) = packed_float4(kval);
            *(threadgroup packed_float4 *) (Vsh + jj*LD + dd4*4) = packed_float4(vval);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sfrag[NCB];
        #pragma clang loop unroll(full)
        for (short cb = 0; cb < NCB; ++cb) {
            sfrag[cb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
        #pragma clang loop unroll(full)
        for (short c = 0; c < NV; ++c) {
            #pragma clang loop unroll(full)
            for (short cb = 0; cb < NCB; ++cb) {
                simdgroup_float8x8 mkt;
                simdgroup_load(mkt, Ksh + cb*8*LD + c*8, LD, 0, true);
                simdgroup_multiply_accumulate(sfrag[cb], mq[c], mkt, sfrag[cb]);
            }
        }
        #pragma clang loop unroll(full)
        for (short cb = 0; cb < NCB; ++cb) {
            simdgroup_store(sfrag[cb], ssh_w + cb*8, NCP);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sv[CPL];
        float tmax = -INFINITY;
        for (short c = 0; c < CPL; ++c) {
            const int64_t j = j0 + cq + c;
            float sc = -INFINITY;
            if (i_row < S && j < args.S_kv) {
                const float mv = flash_attn_train_mask_val_t<CAUSAL>(args.causal_prefix, mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i_row, j);
                if (mv != -INFINITY) {
                    sc = args.scale*ssh_w[r*NCP + cq + c] + mv;
                }
            }
            sv[c] = sc;
            tmax = max(tmax, sc);
        }
        tmax = max(tmax, simd_shuffle(tmax, lane ^ 1));
        tmax = max(tmax, simd_shuffle(tmax, lane ^ 2));
        const float m_new = max(mrun, tmax);
        float ps = 0.0f;
        float pv[CPL];
        for (short c = 0; c < CPL; ++c) {
            pv[c] = (sv[c] == -INFINITY || m_new == -INFINITY) ? 0.0f : exp(sv[c] - m_new);
            ps += pv[c];
        }
        ps += simd_shuffle(ps, lane ^ 1);
        ps += simd_shuffle(ps, lane ^ 2);
        float corr = 1.0f;
        if (m_new != -INFINITY) {
            corr = (mrun == -INFINITY) ? 0.0f : exp(mrun - m_new);
            lrun = lrun*corr + ps;
            mrun = m_new;
        }
        for (short c = 0; c < CPL; ++c) {
            ssh_w[r*NCP + cq + c] = pv[c];   // P overwrites S (each lane its own elements)
        }
        if ((lane & 3) == 0) {
            dsh_w[r*8 + r] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 pmat[NCB];
        #pragma clang loop unroll(full)
        for (short cb = 0; cb < NCB; ++cb) {
            simdgroup_load(pmat[cb], ssh_w + cb*8, NCP);
        }
        simdgroup_float8x8 dmat;
        simdgroup_load(dmat, dsh_w, 8);
        #pragma clang loop unroll(full)
        for (short c = 0; c < NV; ++c) {
            simdgroup_float8x8 t8;
            simdgroup_multiply_accumulate(t8, dmat, acc[c], zero8);   // diag(corr) * acc
            #pragma clang loop unroll(full)
            for (short cb = 0; cb < NCB; ++cb) {
                simdgroup_float8x8 mv8;
                simdgroup_load(mv8, Vsh + cb*8*LD + c*8, LD);
                simdgroup_multiply_accumulate(t8, pmat[cb], mv8, t8);
            }
            acc[c] = t8;
        }
    }

    const bool  row_dead = !(lrun > 0.0f);
    const float lse_r    = row_dead ? 0.0f : (mrun + log(lrun));
    const float inv_l    = row_dead ? 0.0f : 1.0f/lrun;

    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (short c = 0; c < NV; ++c) {
            simdgroup_store(acc[c], ssh_w, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (i_row < S) {
                const int64_t  row  = h + Nh*(i_row + S*b);
                device float * orow = (device float *) (dst + row*D*sizeof(float));
                const short    cc   = (lane & 3)*2;
                orow[c*8 + cc    ] = row_dead ? 0.0f : ssh_w[r*8 + cc    ]*inv_l;
                orow[c*8 + cc + 1] = row_dead ? 0.0f : ssh_w[r*8 + cc + 1]*inv_l;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (i_row < S && (lane & 3) == 0) {
            const int64_t  row = h + Nh*(i_row + S*b);
            device float * lse = (device float *) (dst + args.offs_lse);
            lse[row] = lse_r;
        }
    }
}

typedef decltype(kernel_flash_attn_train_mm3_1p_f32<64, false>) kernel_flash_attn_train_mm3_1p_f32_t;
template [[host_name("kernel_flash_attn_train_mm3_1p_n4_c8_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, false, 4, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n4_c8_causal_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, true, 4, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n4_c8_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, false, 4, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n4_c8_causal_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, true, 4, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n4_c16_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, false, 4, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n4_c16_causal_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, true, 4, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n4_c16_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, false, 4, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n4_c16_causal_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, true, 4, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n6_c8_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, false, 6, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n6_c8_causal_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, true, 6, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n6_c8_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, false, 6, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n6_c8_causal_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, true, 6, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n6_c16_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, false, 6, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n6_c16_causal_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, true, 6, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n6_c16_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, false, 6, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n6_c16_causal_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, true, 6, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n8_c8_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, false, 8, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n8_c8_causal_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, true, 8, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n8_c8_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, false, 8, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n8_c8_causal_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, true, 8, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n8_c16_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, false, 8, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n8_c16_causal_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, true, 8, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n8_c16_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, false, 8, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n8_c16_causal_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, true, 8, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n12_c8_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, false, 12, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n12_c8_causal_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, true, 12, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n12_c8_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, false, 12, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n12_c8_causal_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, true, 12, 8>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n12_c16_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, false, 12, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n12_c16_causal_f32_d64")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<64, true, 12, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n12_c16_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, false, 12, 16>;
template [[host_name("kernel_flash_attn_train_mm3_1p_n12_c16_causal_f32_d128")]] kernel kernel_flash_attn_train_mm3_1p_f32_t kernel_flash_attn_train_mm3_1p_f32<128, true, 12, 16>;

// Backward, precompute: delta[r] = sum_d dO[d,r] * O[d,r], r = h + Nh*(i +
// S*b) over every (query row, head, batch) triple, Nh*S*Bn rows total. Read
// back by both backward passes below instead of each recomputing it --
// matches CUDA's fa_train_bwd_delta_f32 exactly. One simdgroup per row, no
// threadgroup memory, no barriers (nothing here is shared across rows).
template <int D>
kernel void kernel_flash_attn_train_delta_f32(
        constant ggml_metal_kargs_flash_attn_train_back & args,
        device const char * fwd,   // packed O|LSE (only the O region is read)
        device const char * dfwd,  // packed dO|(LSE region ignored, spec 9.7)
        device       char * delta, // [Nh*S*Bn] F32, this op's own extra scratch region
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NV = D/N_SIMDWIDTH;

    const short   lane  = tiisg;
    const int64_t nrows = args.Nh*args.S*args.Bn;
    const int64_t r     = tgpig.x*HOT_STEP_FA_TRAIN_NSG + sgitg;
    if (r >= nrows) {
        // r is simdgroup-uniform, so the whole simdgroup leaves together and
        // simd_sum below never runs with a partially exited simdgroup.
        return;
    }

    device const float * orow = (device const float *) (fwd  + r*D*sizeof(float));
    device const float * drow = (device const float *) (dfwd + r*D*sizeof(float));
    device       float * out  = (device       float *) delta;

    float s = 0.0f;
    for (short c = 0; c < NV; ++c) {
        s += orow[lane + N_SIMDWIDTH*c]*drow[lane + N_SIMDWIDTH*c];
    }
    s = simd_sum(s);
    if (lane == 0) {
        out[r] = s;
    }
}

typedef decltype(kernel_flash_attn_train_delta_f32<64>) kernel_flash_attn_train_delta_f32_t;
template [[host_name("kernel_flash_attn_train_delta_f32_d64" )]] kernel kernel_flash_attn_train_delta_f32_t kernel_flash_attn_train_delta_f32<64>;
template [[host_name("kernel_flash_attn_train_delta_f32_d128")]] kernel kernel_flash_attn_train_delta_f32_t kernel_flash_attn_train_delta_f32<128>;

// Backward, pass A: dQ. Grid (ceil(S/HOT_STEP_FA_TRAIN_BNSG), Nh, Bn) -- same
// shape as the forward. Recomputes S(i,j)/P(i,j) from Q/K/V and the
// forward's own stored LSE (never a freshly-computed softmax normaliser --
// reusing LSE is what keeps this numerically consistent with the forward
// that produced it; see ggml_flash_attn_train_set_prec's own comment on
// precision being a property of the op PAIR, ggml.c). di is the precomputed
// delta[] value above, not recomputed here.
template <int D>
kernel void kernel_flash_attn_train_back_dq_f32(
        constant ggml_metal_kargs_flash_attn_train_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * fwd,   // packed O|LSE (only the LSE region is read)
        device const char * dfwd,  // packed dO|(LSE region ignored, spec 9.7)
        device const char * delta, // [Nh*S*Bn] F32, from kernel_flash_attn_train_delta_f32
        device       char * dst,   // packed dQ|dK|dV
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NV = D/N_SIMDWIDTH;
    constexpr short LD = D + 1;

    threadgroup float Ksh [HOT_STEP_FA_TRAIN_BK*LD];
    threadgroup float Vsh [HOT_STEP_FA_TRAIN_BK*LD];
    threadgroup float Qsh [HOT_STEP_FA_TRAIN_BNSG*LD];
    threadgroup float dOsh[HOT_STEP_FA_TRAIN_BNSG*LD];
    threadgroup int   live_sh[HOT_STEP_FA_TRAIN_BNSG];

    const short  lane = tiisg;
    const short  w    = sgitg;
    const ushort tid  = w*N_SIMDWIDTH + lane;

    // Lane-pair split: HOT_STEP_FA_TRAIN_BK (16, the K/V/Q tile's row count)
    // is exactly half of N_SIMDWIDTH (32), so pairing adjacent lanes lets
    // both halves of each simdgroup do useful work -- each pair of lanes
    // (2*trow, 2*trow+1) cooperates on tile row `trow`, splitting its
    // D-length dot product across `thalf` and recombining with
    // simd_shuffle(x, lane ^ 1). See the plan doc's "lane-pair-split"
    // section for the full rationale.
    static_assert(HOT_STEP_FA_TRAIN_BK * 2 == N_SIMDWIDTH, "lane-pair split assumes BK == N_SIMDWIDTH/2");
    const short  trow  = lane >> 1;
    const short  thalf = lane &  1;

    const int64_t Nh = args.Nh;
    const int64_t S  = args.S;

    const int64_t i  = tgpig.x*HOT_STEP_FA_TRAIN_BNSG + w;
    const int64_t h  = tgpig.y;
    const int64_t b  = tgpig.z;
    const int64_t hk = h/args.G;

    const bool active = i < S;

    float lse_i = 0.0f;
    float del_i = 0.0f;
    if (active) {
        device const float * qrow = (device const float *) (q + i*args.nb01 + h*args.nb02 + b*args.nb03);
        const int64_t         row = h + Nh*(i + S*b);
        device const float * drow = (device const float *) (dfwd + row*D*sizeof(float));
        for (short c = 0; c < NV; ++c) {
            Qsh [w*LD + lane + N_SIMDWIDTH*c] = qrow[lane + N_SIMDWIDTH*c];
            dOsh[w*LD + lane + N_SIMDWIDTH*c] = drow[lane + N_SIMDWIDTH*c];
        }
        device const float * lse_arr = (device const float *) (fwd + args.offs_lse);
        device const float * delta_f = (device const float *) delta;
        lse_i = lse_arr[row];
        del_i = delta_f[row];
    }
    // Qsh/dOsh rows are read back only by the simdgroup that wrote them.
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float dq_acc[NV];
    for (short c = 0; c < NV; ++c) {
        dq_acc[c] = 0.0f;
    }

    device const half * mp = (device const half *) mask;

    for (int64_t j0 = 0; j0 < args.S_kv; j0 += HOT_STEP_FA_TRAIN_BK) {
        // (A) previous tile's compute is done: safe to overwrite K/V and flags
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int64_t j   = j0 + trow;
        // Both lanes of a pair (2*trow, 2*trow+1) address the same tile
        // row `trow`, which is inherently < HOT_STEP_FA_TRAIN_BK (see the
        // static_assert above) -- no separate bound needed here.
        const bool    jok = active && (j < args.S_kv);

        float mv = 0.0f;
        if (jok) {
            mv = flash_attn_train_mask_val(mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i, j);
        }
        const bool live = jok && (mv != -INFINITY);

        const bool anyw = simd_any(live);
        if (lane == 0) {
            live_sh[w] = anyw ? 1 : 0;
        }

        // (B) tile liveness known to the whole threadgroup
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool block_live = false;
        for (short t = 0; t < HOT_STEP_FA_TRAIN_BNSG; ++t) {
            block_live = block_live || (live_sh[t] != 0);
        }
        if (!block_live) {
            continue;
        }

        const int64_t nk = (args.S_kv - j0 < HOT_STEP_FA_TRAIN_BK) ? (args.S_kv - j0) : (int64_t) HOT_STEP_FA_TRAIN_BK;
        // Vectorized: one float4 load covers 4 consecutive feature-dim
        // elements (D is a multiple of 4), 4x fewer load-pipeline
        // instructions for the same bytes. packed_float4 (not float4) for
        // the threadgroup store: LD=D+1 staggers each row's start to dodge
        // bank conflicts, so jj*LD isn't always 16B-aligned -- packed_float4
        // has no alignment requirement, float4 there would be UB.
        constexpr short D4 = D/4;
        for (short idx = tid; idx < HOT_STEP_FA_TRAIN_BK*D4; idx += HOT_STEP_FA_TRAIN_BNSG*N_SIMDWIDTH) {
            const short jj  = idx/D4;
            const short dd4 = idx%D4;
            float4 kval = float4(0.0f);
            float4 vval = float4(0.0f);
            if (jj < nk) {
                device const float4 * krow = (device const float4 *) (k + (j0 + jj)*args.nb11 + hk*args.nb12 + b*args.nb13);
                device const float4 * vrow = (device const float4 *) (v + (j0 + jj)*args.nb21 + hk*args.nb22 + b*args.nb23);
                kval = krow[dd4];
                vval = vrow[dd4];
            }
            *(threadgroup packed_float4 *) (Ksh + jj*LD + dd4*4) = packed_float4(kval);
            *(threadgroup packed_float4 *) (Vsh + jj*LD + dd4*4) = packed_float4(vval);
        }

        // (C) K/V tile staged
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float cf = 0.0f;
        if (live) {
            threadgroup const float * qs = Qsh + w*LD;
            threadgroup const float * ks = Ksh + trow*LD;
            // See the forward kernel's lane-pair-split comment above.
            float dot_half = 0.0f;
            for (short d = thalf; d < D; d += 2) {
                dot_half += qs[d]*ks[d];
            }
            const float dot = dot_half + simd_shuffle(dot_half, lane ^ 1);
            // the mask is additive AFTER the scale, which is why dQ/dK
            // carry a scale factor and dV does not
            const float s = args.scale*dot + mv;
            const float p = exp(s - lse_i);
            if (p != 0.0f) {
                threadgroup const float * os = dOsh + w*LD;
                threadgroup const float * vs = Vsh  + trow*LD;
                float dp_half = 0.0f;
                for (short d = thalf; d < D; d += 2) {
                    dp_half += os[d]*vs[d];
                }
                const float dp = dp_half + simd_shuffle(dp_half, lane ^ 1);
                // dP is finite by construction and the multiply by P kills
                // it where P is 0. Never reordered into (dP - di) times
                // something that can be inf.
                cf = args.scale*(p*(dp - del_i));
            }
        }

        // Fixed source lane (2*r, row r's thalf==0 lane), so the
        // accumulation order is fixed too. Both lanes of a pair computed
        // the identical cf, so either would do -- 2*r is picked for
        // consistency with the forward kernel.
        for (short r = 0; r < HOT_STEP_FA_TRAIN_BK; ++r) {
            const float cr = simd_shuffle(cf, 2*r);
            if (cr == 0.0f) {
                continue;   // dS is exactly 0 here; adding 0*K changes nothing
            }
            threadgroup const float * ks = Ksh + r*LD;
            for (short c = 0; c < NV; ++c) {
                dq_acc[c] += cr*ks[lane + N_SIMDWIDTH*c];
            }
        }
    }

    if (!active) {
        return;
    }

    // written exactly once, by exactly this simdgroup
    const int64_t  dq_row = i + S*(h + Nh*b);
    device float * dqi    = (device float *) (dst + args.offs_dq + dq_row*D*sizeof(float));
    for (short c = 0; c < NV; ++c) {
        dqi[lane + N_SIMDWIDTH*c] = dq_acc[c];
    }
}

typedef decltype(kernel_flash_attn_train_back_dq_f32<64>) kernel_flash_attn_train_back_dq_f32_t;
template [[host_name("kernel_flash_attn_train_back_dq_f32_d64" )]] kernel kernel_flash_attn_train_back_dq_f32_t kernel_flash_attn_train_back_dq_f32<64>;
template [[host_name("kernel_flash_attn_train_back_dq_f32_d128")]] kernel kernel_flash_attn_train_back_dq_f32_t kernel_flash_attn_train_back_dq_f32<128>;

// B3b: dQ with simdgroup_matrix (opt-in, GGML_METAL_FA_TRAIN_MM3_BWD=1).
// Same grid/outputs/contract as kernel_flash_attn_train_back_dq_f32, but per
// 16-key tile and per simdgroup (8 query rows):
//   S  = Q K^T            (mq fragments, K tile staged transposed-loaded)
//   dP = dO V^T           (mdo fragments)
//   P  = exp(scale*S + mask - LSE), dS = scale * P * (dP - delta)   (per lane, 8 rows x 4 cols)
//   dQ += dS K            (acc fragments, dS staged back through the S scratch)
// All threadgroup<->simdgroup handoffs use threadgroup_barrier: simdgroup_barrier
// was NOT sufficient on M1 for the forward (garbage + non-deterministic).
#define HOT_STEP_FA_TRAIN_BDQ_NSG 8    // keep in sync with ggml-metal-ops.cpp
#define HOT_STEP_FA_TRAIN_BDQ_NC  8    // keep in sync with ggml-metal-ops.cpp

template <int D, bool CAUSAL, int NSG_ = 8>
kernel void kernel_flash_attn_train_back_dq_mm_f32(
        constant ggml_metal_kargs_flash_attn_train_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * fwd,
        device const char * dfwd,
        device const char * delta,
        device       char * dst,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = NSG_;
    constexpr short NC  = HOT_STEP_FA_TRAIN_BDQ_NC;
    constexpr short NCB = NC/8;
    constexpr short NCP = NC + HOT_STEP_FA_TRAIN_SPAD;   // padded stride of the S/dS and dP scratch (bank conflicts)
    constexpr short NV  = D/8;
    constexpr short LD  = D + 1;
    constexpr short D4  = D/4;

    // smem: [Ksh NC*LD][Vsh NC*LD][Ssh NSG*8*NC][Psh NSG*8*NC][live NSG ints]
    threadgroup float * Ksh = (threadgroup float *) shmem;
    threadgroup float * Vsh = Ksh + NC*LD;
    threadgroup float * Ssh = Vsh + NC*LD;
    threadgroup float * Psh = Ssh + NSG*8*NCP;
    threadgroup int   * live_sh = (threadgroup int *) (Psh + NSG*8*NCP);

    const short  lane = tiisg;
    const short  w    = sgitg;
    const ushort tid  = w*N_SIMDWIDTH + lane;

    const int64_t Nh = args.Nh;
    const int64_t S  = args.S;

    const int64_t nblk = (args.S + NSG*8 - 1)/(NSG*8);
    const int64_t blk  = CAUSAL ? nblk - 1 - (int64_t) tgpig.x : (int64_t) tgpig.x;
    const int64_t i0 = blk*(NSG*8) + w*8;
    int64_t j_end = args.S_kv;
    if (CAUSAL) {
        const int64_t i_last = min((int64_t) args.S - 1, blk*(NSG*8) + NSG*8 - 1);
        j_end = min((int64_t) args.S_kv, (int64_t) args.causal_prefix + i_last + 1);
    }
    const int64_t h  = tgpig.y;
    const int64_t b  = tgpig.z;
    const int64_t hk = h/args.G;

    threadgroup float * ssh_w = Ssh + w*8*NCP;   // S -> dS
    threadgroup float * psh_w = Psh + w*8*NCP;   // dP

    const short   r     = lane >> 2;
    const short   CPL   = NC/4;           // columns per lane (4 lanes per row)
    const short   cq    = (lane & 3)*CPL;
    const int64_t i_row = i0 + r;

    float lse_r = 0.0f;
    float del_r = 0.0f;
    if (i_row < S) {
        const int64_t row = h + Nh*(i_row + S*b);
        lse_r = ((device const float *) (fwd + args.offs_lse))[row];
        del_r = ((device const float *) delta)[row];
    }

    // ---- stage Q then dO into register fragments (through this simdgroup's slice of the K/V buffer) ----
    simdgroup_float8x8 mq[NV];
    simdgroup_float8x8 mdo[NV];
    {
        // chunked through this simdgroup's slice of the K/V buffer: NPART chunks of CW floats per row
        constexpr short NPART = (NSG*8*(D/2) <= 2*NC*LD) ? 2 : ((NSG*8*(D/4) <= 2*NC*LD) ? 4 : 8);
        constexpr short CW    = D/NPART;
        threadgroup float * qs = Ksh + w*(8*CW);
        #pragma clang loop unroll(full)
        for (short hf = 0; hf < NPART; ++hf) {
            for (short idx = lane; idx < 8*(CW/4); idx += N_SIMDWIDTH) {
                const short rr = idx/(CW/4);
                const short c4 = idx%(CW/4);
                float4 qv = float4(0.0f);
                if (i0 + rr < S) {
                    device const float4 * qrow = (device const float4 *) (q + (i0 + rr)*args.nb01 + h*args.nb02 + b*args.nb03);
                    qv = qrow[hf*(CW/4) + c4];
                }
                *(threadgroup packed_float4 *) (qs + rr*CW + c4*4) = packed_float4(qv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (short c = 0; c < NV/NPART; ++c) {
                simdgroup_load(mq[hf*(NV/NPART) + c], qs + c*8, CW);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (short hf = 0; hf < NPART; ++hf) {
            for (short idx = lane; idx < 8*(CW/4); idx += N_SIMDWIDTH) {
                const short rr = idx/(CW/4);
                const short c4 = idx%(CW/4);
                float4 dv = float4(0.0f);
                if (i0 + rr < S) {
                    const int64_t row = h + Nh*((i0 + rr) + S*b);
                    device const float4 * drow = (device const float4 *) (dfwd + row*D*sizeof(float));
                    dv = drow[hf*(CW/4) + c4];
                }
                *(threadgroup packed_float4 *) (qs + rr*CW + c4*4) = packed_float4(dv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (short c = 0; c < NV/NPART; ++c) {
                simdgroup_load(mdo[hf*(NV/NPART) + c], qs + c*8, CW);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    simdgroup_float8x8 acc[NV];
    #pragma clang loop unroll(full)
    for (short c = 0; c < NV; ++c) {
        acc[c] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    device const half * mp = (device const half *) mask;

    for (int64_t j0 = 0; j0 < j_end; j0 += NC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (args.has_mask) {
            bool live = false;
            if (i_row < S) {
                for (short c = 0; c < CPL; ++c) {
                    const int64_t j = j0 + cq + c;
                    if (j < args.S_kv) {
                        const float mv = flash_attn_train_mask_val_t<CAUSAL>(args.causal_prefix, mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i_row, j);
                        if (mv != -INFINITY) {
                            live = true;
                        }
                    }
                }
            }
            const bool anyw = simd_any(live);
            if (lane == 0) {
                live_sh[w] = anyw ? 1 : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool block_live = false;
            for (short t = 0; t < NSG; ++t) {
                block_live = block_live || (live_sh[t] != 0);
            }
            if (!block_live) {
                continue;
            }
        }

        const int64_t nk = (args.S_kv - j0 < NC) ? (args.S_kv - j0) : (int64_t) NC;

        for (short idx = tid; idx < NC*D4; idx += NSG*N_SIMDWIDTH) {
            const short jj  = idx/D4;
            const short dd4 = idx%D4;
            float4 kval = float4(0.0f);
            float4 vval = float4(0.0f);
            if (jj < nk) {
                device const float4 * krow = (device const float4 *) (k + (j0 + jj)*args.nb11 + hk*args.nb12 + b*args.nb13);
                device const float4 * vrow = (device const float4 *) (v + (j0 + jj)*args.nb21 + hk*args.nb22 + b*args.nb23);
                kval = krow[dd4];
                vval = vrow[dd4];
            }
            *(threadgroup packed_float4 *) (Ksh + jj*LD + dd4*4) = packed_float4(kval);
            *(threadgroup packed_float4 *) (Vsh + jj*LD + dd4*4) = packed_float4(vval);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q K^T and dP = dO V^T (8 rows x NC keys each)
        simdgroup_float8x8 sfrag[NCB];
        simdgroup_float8x8 pfrag[NCB];
        #pragma clang loop unroll(full)
        for (short cb = 0; cb < NCB; ++cb) {
            sfrag[cb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            pfrag[cb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
        #pragma clang loop unroll(full)
        for (short c = 0; c < NV; ++c) {
            #pragma clang loop unroll(full)
            for (short cb = 0; cb < NCB; ++cb) {
                simdgroup_float8x8 mkt;
                simdgroup_float8x8 mvt;
                simdgroup_load(mkt, Ksh + cb*8*LD + c*8, LD, 0, true);
                simdgroup_load(mvt, Vsh + cb*8*LD + c*8, LD, 0, true);
                simdgroup_multiply_accumulate(sfrag[cb], mq[c],  mkt, sfrag[cb]);
                simdgroup_multiply_accumulate(pfrag[cb], mdo[c], mvt, pfrag[cb]);
            }
        }
        #pragma clang loop unroll(full)
        for (short cb = 0; cb < NCB; ++cb) {
            simdgroup_store(sfrag[cb], ssh_w + cb*8, NCP);
            simdgroup_store(pfrag[cb], psh_w + cb*8, NCP);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // per-lane: row r, columns cq..cq+3 -> dS (overwrites S in the scratch)
        for (short c = 0; c < CPL; ++c) {
            const int64_t j = j0 + cq + c;
            float ds = 0.0f;
            if (i_row < S && j < args.S_kv) {
                const float mv = flash_attn_train_mask_val_t<CAUSAL>(args.causal_prefix, mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i_row, j);
                if (mv != -INFINITY) {
                    const float s = args.scale*ssh_w[r*NCP + cq + c] + mv;
                    const float p = exp(s - lse_r);
                    if (p != 0.0f) {
                        ds = args.scale*(p*(psh_w[r*NCP + cq + c] - del_r));
                    }
                }
            }
            ssh_w[r*NCP + cq + c] = ds;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dQ += dS K
        simdgroup_float8x8 dsm[NCB];
        #pragma clang loop unroll(full)
        for (short cb = 0; cb < NCB; ++cb) {
            simdgroup_load(dsm[cb], ssh_w + cb*8, NCP);
        }
        #pragma clang loop unroll(full)
        for (short c = 0; c < NV; ++c) {
            #pragma clang loop unroll(full)
            for (short cb = 0; cb < NCB; ++cb) {
                simdgroup_float8x8 mk8;
                simdgroup_load(mk8, Ksh + cb*8*LD + c*8, LD);
                simdgroup_multiply_accumulate(acc[c], dsm[cb], mk8, acc[c]);
            }
        }
    }

    // ---- write dQ ----
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (short c = 0; c < NV; ++c) {
            simdgroup_store(acc[c], ssh_w, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (i_row < S) {
                const int64_t  dq_row = i_row + S*(h + Nh*b);
                device float * dqi    = (device float *) (dst + args.offs_dq + dq_row*D*sizeof(float));
                const short    cc     = (lane & 3)*2;
                dqi[c*8 + cc    ] = ssh_w[r*8 + cc    ];
                dqi[c*8 + cc + 1] = ssh_w[r*8 + cc + 1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

typedef decltype(kernel_flash_attn_train_back_dq_mm_f32<64, false>) kernel_flash_attn_train_back_dq_mm_f32_t;
template [[host_name("kernel_flash_attn_train_back_dq_mm_f32_d64" )]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<64, false>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_causal_f32_d64" )]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<64, true>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_f32_d128")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<128, false>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_causal_f32_d128")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<128, true>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n4_f32_d64")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<64, false, 4>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n4_causal_f32_d64")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<64, true, 4>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n4_f32_d128")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<128, false, 4>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n4_causal_f32_d128")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<128, true, 4>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n6_f32_d64")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<64, false, 6>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n6_causal_f32_d64")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<64, true, 6>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n6_f32_d128")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<128, false, 6>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n6_causal_f32_d128")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<128, true, 6>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n12_f32_d64")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<64, false, 12>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n12_causal_f32_d64")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<64, true, 12>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n12_f32_d128")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<128, false, 12>;
template [[host_name("kernel_flash_attn_train_back_dq_mm_n12_causal_f32_d128")]] kernel kernel_flash_attn_train_back_dq_mm_f32_t kernel_flash_attn_train_back_dq_mm_f32<128, true, 12>;

// Backward, pass B: dK, dV. Grid (ceil(S_kv/HOT_STEP_FA_TRAIN_BNSG), Nkv,
// Bn); one simdgroup per KV row (j,hk,b), summing the contribution of every
// query row that attends to it across the GQA group's G query heads. Query
// rows and dO are tiled through threadgroup memory HOT_STEP_FA_TRAIN_BK
// (aliased BQ here) at a time and shared by every simdgroup in the
// threadgroup; this simdgroup's own K/V row is staged once, before the
// loop, and stays resident for the whole kernel. Reads only forward-side
// tensors (q,k,v,fwd,dfwd,delta) and writes only the dK/dV region --
// disjoint from pass A's dQ region -- so the two passes need no barrier
// between them (see ggml_metal_op_flash_attn_train_back's own comment).
template <int D>
kernel void kernel_flash_attn_train_back_dkdv_f32(
        constant ggml_metal_kargs_flash_attn_train_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * fwd,
        device const char * dfwd,
        device const char * delta,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NV = D/N_SIMDWIDTH;
    constexpr short LD = D + 1;
    constexpr short BQ = HOT_STEP_FA_TRAIN_BK; // far-side (query) rows staged per tile

    threadgroup float Qsh   [BQ*LD];
    threadgroup float dOsh  [BQ*LD];
    threadgroup float Ksh   [HOT_STEP_FA_TRAIN_BNSG*LD];
    threadgroup float Vsh   [HOT_STEP_FA_TRAIN_BNSG*LD];
    threadgroup float lse_sh[BQ];
    threadgroup float del_sh[BQ];
    threadgroup int   live_sh[HOT_STEP_FA_TRAIN_BNSG];

    const short  lane = tiisg;
    const short  w    = sgitg;
    const ushort tid  = w*N_SIMDWIDTH + lane;

    // Lane-pair split: HOT_STEP_FA_TRAIN_BK (16, the K/V/Q tile's row count)
    // is exactly half of N_SIMDWIDTH (32), so pairing adjacent lanes lets
    // both halves of each simdgroup do useful work -- each pair of lanes
    // (2*trow, 2*trow+1) cooperates on tile row `trow`, splitting its
    // D-length dot product across `thalf` and recombining with
    // simd_shuffle(x, lane ^ 1). See the plan doc's "lane-pair-split"
    // section for the full rationale.
    static_assert(HOT_STEP_FA_TRAIN_BK * 2 == N_SIMDWIDTH, "lane-pair split assumes BK == N_SIMDWIDTH/2");
    const short  trow  = lane >> 1;
    const short  thalf = lane &  1;

    const int64_t Nh   = args.Nh;
    const int64_t S    = args.S;
    const int64_t S_kv = args.S_kv;

    const int64_t j0b = tgpig.x*HOT_STEP_FA_TRAIN_BNSG; // first kv row of the threadgroup
    const int64_t j   = j0b + w;                       // this simdgroup's kv row
    const int64_t hk  = tgpig.y;
    const int64_t b   = tgpig.z;

    const bool jok = j < S_kv;

    // Detached-prefix rows: their dK/dV are never consumed (see
    // ggml_flash_attn_train_set_kv_grad_start), so a tile that lies entirely
    // below kv_grad_start writes zeros and leaves. The condition depends only
    // on tgpig.x, so the whole threadgroup exits together, before any
    // threadgroup_barrier(). A tile straddling the boundary is computed normally.
    if (j0b + HOT_STEP_FA_TRAIN_BNSG <= (int64_t) args.kv_grad_start) {
        if (jok) {
            const int64_t  zrow = j + S_kv*(hk + args.Nkv*b);
            device float * zk   = (device float *) (dst + args.offs_dk + zrow*D*sizeof(float));
            device float * zv   = (device float *) (dst + args.offs_dv + zrow*D*sizeof(float));
            for (short c = 0; c < NV; ++c) {
                zk[lane + N_SIMDWIDTH*c] = 0.0f;
                zv[lane + N_SIMDWIDTH*c] = 0.0f;
            }
        }
        return;
    }

    // The threadgroup's HOT_STEP_FA_TRAIN_BNSG K/V rows are staged once and
    // stay resident. Published by the first threadgroup_barrier() inside
    // the loop below.
    constexpr short D4 = D/4;
    for (short idx = tid; idx < HOT_STEP_FA_TRAIN_BNSG*D4; idx += HOT_STEP_FA_TRAIN_BNSG*N_SIMDWIDTH) {
        const short   jj   = idx/D4;
        const short   dd4  = idx%D4;
        const int64_t jrow = j0b + jj;
        float4 kval = float4(0.0f);
        float4 vval = float4(0.0f);
        if (jrow < S_kv) {
            device const float4 * krow = (device const float4 *) (k + jrow*args.nb11 + hk*args.nb12 + b*args.nb13);
            device const float4 * vrow = (device const float4 *) (v + jrow*args.nb21 + hk*args.nb22 + b*args.nb23);
            kval = krow[dd4];
            vval = vrow[dd4];
        }
        *(threadgroup packed_float4 *) (Ksh + jj*LD + dd4*4) = packed_float4(kval);
        *(threadgroup packed_float4 *) (Vsh + jj*LD + dd4*4) = packed_float4(vval);
    }

    float dk_acc[NV];
    float dv_acc[NV];
    for (short c = 0; c < NV; ++c) {
        dk_acc[c] = 0.0f;
        dv_acc[c] = 0.0f;
    }

    device const half  * mp      = (device const half  *) mask;
    device const float * lse_arr = (device const float *) (fwd + args.offs_lse);
    device const float * delta_f = (device const float *) delta;

    // Deterministic order: query heads ascending, then query tiles
    // ascending -- the same visitation order ggml-cpu/ops.cpp's own pass B
    // uses (its BQ tiling doesn't change the order, only the batching).
    for (int64_t g = 0; g < args.G; ++g) {
        const int64_t h = hk*args.G + g;

        for (int64_t i0 = 0; i0 < S; i0 += BQ) {
            // (A) the previous tile's reads of Qsh/dOsh are done, and on
            //     the very first pass this publishes the K/V staging above
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const int64_t i   = i0 + trow;
            // Both lanes of a pair (2*trow, 2*trow+1) address the same
            // query-tile row `trow`, which is inherently < BQ (== 16, see
            // the static_assert above) -- no separate bound needed here.
            const bool    iok = jok && (i < S);

            float mv = 0.0f;
            if (iok) {
                mv = flash_attn_train_mask_val(mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i, j);
            }
            const bool live = iok && (mv != -INFINITY);

            const bool anyw = simd_any(live);
            if (lane == 0) {
                live_sh[w] = anyw ? 1 : 0;
            }

            // (B) tile liveness known to the whole threadgroup
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool block_live = false;
            for (short t = 0; t < HOT_STEP_FA_TRAIN_BNSG; ++t) {
                block_live = block_live || (live_sh[t] != 0);
            }
            if (!block_live) {
                // block_live is threadgroup-uniform, so every thread takes
                // this and the threadgroup_barrier() at (A) stays uniform.
                continue;
            }

            const int64_t nq = (S - i0 < BQ) ? (S - i0) : (int64_t) BQ;
            constexpr short D4q = D/4;
            for (short idx = tid; idx < BQ*D4q; idx += HOT_STEP_FA_TRAIN_BNSG*N_SIMDWIDTH) {
                const short ii  = idx/D4q;
                const short dd4 = idx%D4q;
                float4 qval = float4(0.0f);
                float4 oval = float4(0.0f);
                if (ii < nq) {
                    device const float4 * qrow = (device const float4 *) (q + (i0 + ii)*args.nb01 + h*args.nb02 + b*args.nb03);
                    device const float4 * drow = (device const float4 *) (dfwd + (h + Nh*((i0 + ii) + S*b))*D*sizeof(float));
                    qval = qrow[dd4];
                    oval = drow[dd4];
                }
                *(threadgroup packed_float4 *) (Qsh  + ii*LD + dd4*4) = packed_float4(qval);
                *(threadgroup packed_float4 *) (dOsh + ii*LD + dd4*4) = packed_float4(oval);
            }
            for (short idx = tid; idx < BQ; idx += HOT_STEP_FA_TRAIN_BNSG*N_SIMDWIDTH) {
                float lv = 0.0f;
                float dv = 0.0f;
                if (idx < nq) {
                    const int64_t r = h + Nh*((i0 + idx) + S*b);
                    lv = lse_arr[r];
                    dv = delta_f[r];
                }
                lse_sh[idx] = lv;
                del_sh[idx] = dv;
            }

            // (C) query tile staged
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float p  = 0.0f;
            float cf = 0.0f;
            if (live) {
                threadgroup const float * qs = Qsh + trow*LD;
                threadgroup const float * ks = Ksh + w*LD;
                // See the forward kernel's lane-pair-split comment above.
                float dot_half = 0.0f;
                for (short d = thalf; d < D; d += 2) {
                    dot_half += qs[d]*ks[d];
                }
                const float dot = dot_half + simd_shuffle(dot_half, lane ^ 1);
                const float s = args.scale*dot + mv;
                p = exp(s - lse_sh[trow]);
                if (p != 0.0f) {
                    threadgroup const float * os = dOsh + trow*LD;
                    threadgroup const float * vs = Vsh  + w*LD;
                    float dp_half = 0.0f;
                    for (short d = thalf; d < D; d += 2) {
                        dp_half += os[d]*vs[d];
                    }
                    const float dp = dp_half + simd_shuffle(dp_half, lane ^ 1);
                    cf = args.scale*(p*(dp - del_sh[trow]));
                }
            }

            // Fixed source lane (2*r, row r's thalf==0 lane), so the
            // accumulation order is fixed too. Both lanes of a pair
            // computed the identical p/cf.
            for (short r = 0; r < BQ; ++r) {
                const float pr = simd_shuffle(p, 2*r);
                if (pr == 0.0f) {
                    continue;   // masked, or vanished: contributes exactly nothing
                }
                const float cr = simd_shuffle(cf, 2*r);
                threadgroup const float * os = dOsh + r*LD;
                threadgroup const float * qs = Qsh  + r*LD;
                for (short c = 0; c < NV; ++c) {
                    dv_acc[c] += pr*os[lane + N_SIMDWIDTH*c];
                    dk_acc[c] += cr*qs[lane + N_SIMDWIDTH*c];
                }
            }
        }
    }

    if (!jok) {
        return;
    }

    // written exactly once, by exactly this simdgroup
    const int64_t  dkv_row = j + S_kv*(hk + args.Nkv*b);
    device float * dkj     = (device float *) (dst + args.offs_dk + dkv_row*D*sizeof(float));
    device float * dvj     = (device float *) (dst + args.offs_dv + dkv_row*D*sizeof(float));
    for (short c = 0; c < NV; ++c) {
        dkj[lane + N_SIMDWIDTH*c] = dk_acc[c];
        dvj[lane + N_SIMDWIDTH*c] = dv_acc[c];
    }
}

typedef decltype(kernel_flash_attn_train_back_dkdv_f32<64>) kernel_flash_attn_train_back_dkdv_f32_t;
template [[host_name("kernel_flash_attn_train_back_dkdv_f32_d64" )]] kernel kernel_flash_attn_train_back_dkdv_f32_t kernel_flash_attn_train_back_dkdv_f32<64>;
template [[host_name("kernel_flash_attn_train_back_dkdv_f32_d128")]] kernel kernel_flash_attn_train_back_dkdv_f32_t kernel_flash_attn_train_back_dkdv_f32<128>;

// B3c: dV / dK with simdgroup_matrix (opt-in, GGML_METAL_FA_TRAIN_MM3_BWD_KV=1).
// Two kernels (DK = false: dV, DK = true: dK) instead of one, to keep the
// fragment count <= 48 (K, [V,] accumulator). One simdgroup owns 8 KV rows
// (j, hk, b) and visits, per GQA head g ascending, the query rows in tiles of
// 16 (NQ). Per tile, rows = keys, columns = queries ("transposed" orientation):
//   S^T  = K Q^T                      (mk fragments in registers, Q tile transposed-loaded)
//   dV  += P^T dO                     with P^T = exp(scale*S^T + mask - LSE_i)
//   dP^T = V dO^T, dS^T = scale*P^T*(dP^T - delta_i), dK += dS^T Q   (DK only; mv fragments)
// Same masked-tile skip, kv_grad_start early-out and accumulation order
// (g ascending, query tiles ascending) as the scalar kernel.
#define HOT_STEP_FA_TRAIN_BKV_NSG 8    // keep in sync with ggml-metal-ops.cpp
#define HOT_STEP_FA_TRAIN_BKV_NQ  8    // keep in sync with ggml-metal-ops.cpp

template <int D, bool DK, bool CAUSAL, int NSG_ = 8>
kernel void kernel_flash_attn_train_back_kv_mm_f32(
        constant ggml_metal_kargs_flash_attn_train_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * fwd,
        device const char * dfwd,
        device const char * delta,
        device       char * dst,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = NSG_;
    constexpr short NQ  = HOT_STEP_FA_TRAIN_BKV_NQ;
    constexpr short NQB = NQ/8;
    constexpr short NQP = NQ + HOT_STEP_FA_TRAIN_SPAD;   // padded stride of the S^T/dS^T and dP^T scratch
    constexpr short NV  = D/8;
    constexpr short LD  = D + 1;
    constexpr short D4  = D/4;

    // smem: [Qsh NQ*LD][dOsh NQ*LD][Ssh NSG*8*NQ][Psh NSG*8*NQ (DK only)][lse NQ][del NQ][live NSG ints]
    threadgroup float * Qsh  = (threadgroup float *) shmem;
    threadgroup float * dOsh = Qsh + NQ*LD;
    threadgroup float * Ssh  = dOsh + NQ*LD;
    threadgroup float * Psh  = Ssh + NSG*8*NQP;
    threadgroup float * lse_sh = Psh + (DK ? NSG*8*NQP : 0);
    threadgroup float * del_sh = lse_sh + NQ;
    threadgroup int   * live_sh = (threadgroup int *) (del_sh + NQ);

    const short  lane = tiisg;
    const short  w    = sgitg;
    const ushort tid  = w*N_SIMDWIDTH + lane;

    const int64_t Nh   = args.Nh;
    const int64_t S    = args.S;
    const int64_t S_kv = args.S_kv;

    const int64_t j0b = tgpig.x*(NSG*8);       // first kv row of the threadgroup
    const int64_t j0  = j0b + w*8;             // first kv row of this simdgroup
    const int64_t hk  = tgpig.y;
    const int64_t b   = tgpig.z;

    // detached-prefix tile: write zeros and leave (threadgroup-uniform, before any barrier)
    if (j0b + NSG*8 <= (int64_t) args.kv_grad_start) {
        for (short idx = tid; idx < NSG*8*D; idx += NSG*N_SIMDWIDTH) {
            const int64_t jr = j0b + idx/D;
            if (jr < S_kv) {
                const int64_t zrow = jr + S_kv*(hk + args.Nkv*b);
                device float * z = (device float *) (dst + (DK ? args.offs_dk : args.offs_dv) + zrow*D*sizeof(float));
                z[idx%D] = 0.0f;
            }
        }
        return;
    }

    threadgroup float * ssh_w = Ssh + w*8*NQP;   // S^T -> P^T or dS^T
    threadgroup float * psh_w = Psh + w*8*NQP;   // dP^T (DK only)

    const short   r     = lane >> 2;
    const short   CPL   = NQ/4;           // query columns per lane (4 lanes per row)
    const short   cq    = (lane & 3)*CPL;
    const int64_t j_row = j0 + r;               // this lane's kv row

    // ---- stage K (and V for DK) into register fragments through the Q/dO tile buffer ----
    simdgroup_float8x8 mk[NV];
    simdgroup_float8x8 mv[DK ? NV : 1];
    {
        // chunked: NPART chunks of CW floats per row, NSG*8*CW <= 2*NQ*LD
        constexpr short NPART = (NSG*8*(D/2) <= 2*NQ*LD) ? 2 : ((NSG*8*(D/4) <= 2*NQ*LD) ? 4 : 8);
        constexpr short CW    = D/NPART;
        threadgroup float * ks = Qsh + w*(8*CW);
        #pragma clang loop unroll(full)
        for (short hf = 0; hf < NPART; ++hf) {
            for (short idx = lane; idx < 8*(CW/4); idx += N_SIMDWIDTH) {
                const short rr = idx/(CW/4);
                const short c4 = idx%(CW/4);
                float4 kv4 = float4(0.0f);
                if (j0 + rr < S_kv) {
                    device const float4 * krow = (device const float4 *) (k + (j0 + rr)*args.nb11 + hk*args.nb12 + b*args.nb13);
                    kv4 = krow[hf*(CW/4) + c4];
                }
                *(threadgroup packed_float4 *) (ks + rr*CW + c4*4) = packed_float4(kv4);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (short c = 0; c < NV/NPART; ++c) {
                simdgroup_load(mk[hf*(NV/NPART) + c], ks + c*8, CW);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (DK) {
            #pragma clang loop unroll(full)
            for (short hf = 0; hf < NPART; ++hf) {
                for (short idx = lane; idx < 8*(CW/4); idx += N_SIMDWIDTH) {
                    const short rr = idx/(CW/4);
                    const short c4 = idx%(CW/4);
                    float4 vv4 = float4(0.0f);
                    if (j0 + rr < S_kv) {
                        device const float4 * vrow = (device const float4 *) (v + (j0 + rr)*args.nb21 + hk*args.nb22 + b*args.nb23);
                        vv4 = vrow[hf*(CW/4) + c4];
                    }
                    *(threadgroup packed_float4 *) (ks + rr*CW + c4*4) = packed_float4(vv4);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma clang loop unroll(full)
                for (short c = 0; c < NV/NPART; ++c) {
                    simdgroup_load(mv[hf*(NV/NPART) + c], ks + c*8, CW);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    simdgroup_float8x8 acc[NV];
    #pragma clang loop unroll(full)
    for (short c = 0; c < NV; ++c) {
        acc[c] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    device const half  * mp      = (device const half  *) mask;
    device const float * lse_arr = (device const float *) (fwd + args.offs_lse);
    device const float * delta_f = (device const float *) delta;

    // B6: with a causal hint, query rows i < j0b - prefix cannot see any key of this block.
    int64_t i_lo = 0;
    if (CAUSAL) {
        const int64_t first = j0b - (int64_t) args.causal_prefix;
        i_lo = first > 0 ? (first/NQ)*NQ : 0;
    }

    for (int64_t g = 0; g < args.G; ++g) {
        const int64_t h = hk*args.G + g;

        for (int64_t i0 = i_lo; i0 < S; i0 += NQ) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (args.has_mask) {
                bool live = false;
                if (j_row < S_kv) {
                    for (short c = 0; c < CPL; ++c) {
                        const int64_t i = i0 + cq + c;
                        if (i < S) {
                            const float mvv = flash_attn_train_mask_val_t<CAUSAL>(args.causal_prefix, mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i, j_row);
                            if (mvv != -INFINITY) {
                                live = true;
                            }
                        }
                    }
                }
                const bool anyw = simd_any(live);
                if (lane == 0) {
                    live_sh[w] = anyw ? 1 : 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                bool block_live = false;
                for (short t = 0; t < NSG; ++t) {
                    block_live = block_live || (live_sh[t] != 0);
                }
                if (!block_live) {
                    continue;
                }
            }

            const int64_t nq = (S - i0 < NQ) ? (S - i0) : (int64_t) NQ;
            for (short idx = tid; idx < NQ*D4; idx += NSG*N_SIMDWIDTH) {
                const short ii  = idx/D4;
                const short dd4 = idx%D4;
                float4 qval = float4(0.0f);
                float4 oval = float4(0.0f);
                if (ii < nq) {
                    device const float4 * qrow = (device const float4 *) (q + (i0 + ii)*args.nb01 + h*args.nb02 + b*args.nb03);
                    device const float4 * drow = (device const float4 *) (dfwd + (h + Nh*((i0 + ii) + S*b))*D*sizeof(float));
                    qval = qrow[dd4];
                    oval = drow[dd4];
                }
                *(threadgroup packed_float4 *) (Qsh  + ii*LD + dd4*4) = packed_float4(qval);
                *(threadgroup packed_float4 *) (dOsh + ii*LD + dd4*4) = packed_float4(oval);
            }
            for (short idx = tid; idx < NQ; idx += NSG*N_SIMDWIDTH) {
                float lv = 0.0f;
                float dv = 0.0f;
                if (idx < nq) {
                    const int64_t rr = h + Nh*((i0 + idx) + S*b);
                    lv = lse_arr[rr];
                    dv = delta_f[rr];
                }
                lse_sh[idx] = lv;
                del_sh[idx] = dv;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // S^T = K Q^T (and dP^T = V dO^T): 8 keys x NQ queries
            simdgroup_float8x8 sfrag[NQB];
            simdgroup_float8x8 pfrag[DK ? NQB : 1];
            #pragma clang loop unroll(full)
            for (short cb = 0; cb < NQB; ++cb) {
                sfrag[cb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            if (DK) {
                #pragma clang loop unroll(full)
                for (short cb = 0; cb < NQB; ++cb) {
                    pfrag[cb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                }
            }
            #pragma clang loop unroll(full)
            for (short c = 0; c < NV; ++c) {
                #pragma clang loop unroll(full)
                for (short cb = 0; cb < NQB; ++cb) {
                    simdgroup_float8x8 mqt;
                    simdgroup_load(mqt, Qsh + cb*8*LD + c*8, LD, 0, true);
                    simdgroup_multiply_accumulate(sfrag[cb], mk[c], mqt, sfrag[cb]);
                    if (DK) {
                        simdgroup_float8x8 mot;
                        simdgroup_load(mot, dOsh + cb*8*LD + c*8, LD, 0, true);
                        simdgroup_multiply_accumulate(pfrag[cb], mv[DK ? c : 0], mot, pfrag[cb]);
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (short cb = 0; cb < NQB; ++cb) {
                simdgroup_store(sfrag[cb], ssh_w + cb*8, NQP);
                if (DK) {
                    simdgroup_store(pfrag[DK ? cb : 0], psh_w + cb*8, NQP);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // per lane: key row r, query columns cq..cq+3 -> P^T (dV) or dS^T (dK), in place
            for (short c = 0; c < CPL; ++c) {
                const int64_t i = i0 + cq + c;
                float outv = 0.0f;
                if (j_row < S_kv && i < S) {
                    const float mvv = flash_attn_train_mask_val_t<CAUSAL>(args.causal_prefix, mp, args.has_mask, args.mne0, args.mne1, args.mne2, args.mne3, h, b, i, j_row);
                    if (mvv != -INFINITY) {
                        const float s = args.scale*ssh_w[r*NQP + cq + c] + mvv;
                        const float p = exp(s - lse_sh[cq + c]);
                        if (DK) {
                            if (p != 0.0f) {
                                outv = args.scale*(p*(psh_w[r*NQP + cq + c] - del_sh[cq + c]));
                            }
                        } else {
                            outv = p;
                        }
                    }
                }
                ssh_w[r*NQP + cq + c] = outv;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // acc += X^T . (dO for dV, Q for dK)
            simdgroup_float8x8 xm[NQB];
            #pragma clang loop unroll(full)
            for (short cb = 0; cb < NQB; ++cb) {
                simdgroup_load(xm[cb], ssh_w + cb*8, NQP);
            }
            threadgroup float * Bsh = DK ? Qsh : dOsh;
            #pragma clang loop unroll(full)
            for (short c = 0; c < NV; ++c) {
                #pragma clang loop unroll(full)
                for (short cb = 0; cb < NQB; ++cb) {
                    simdgroup_float8x8 mb8;
                    simdgroup_load(mb8, Bsh + cb*8*LD + c*8, LD);
                    simdgroup_multiply_accumulate(acc[c], xm[cb], mb8, acc[c]);
                }
            }
        }
    }

    // ---- write dK or dV (one 8x8 chunk at a time) ----
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (short c = 0; c < NV; ++c) {
            simdgroup_store(acc[c], ssh_w, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (j_row < S_kv) {
                const int64_t  dkv_row = j_row + S_kv*(hk + args.Nkv*b);
                device float * dj      = (device float *) (dst + (DK ? args.offs_dk : args.offs_dv) + dkv_row*D*sizeof(float));
                const short    cc      = (lane & 3)*2;
                dj[c*8 + cc    ] = ssh_w[r*8 + cc    ];
                dj[c*8 + cc + 1] = ssh_w[r*8 + cc + 1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

typedef decltype(kernel_flash_attn_train_back_kv_mm_f32<64, false, false>) kernel_flash_attn_train_back_kv_mm_f32_t;
template [[host_name("kernel_flash_attn_train_back_dv_mm_f32_d64" )]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64,  false, false>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, false, false>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_f32_d64" )]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64,  true,  false>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, true,  false>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_causal_f32_d64" )]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64,  false, true>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_causal_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, false, true>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_causal_f32_d64" )]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64,  true,  true>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_causal_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, true,  true>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n4_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, false, false, 4>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n4_causal_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, false, true, 4>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n4_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, true, false, 4>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n4_causal_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, true, true, 4>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n4_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, false, false, 4>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n4_causal_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, false, true, 4>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n4_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, true, false, 4>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n4_causal_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, true, true, 4>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n6_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, false, false, 6>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n6_causal_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, false, true, 6>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n6_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, true, false, 6>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n6_causal_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, true, true, 6>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n6_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, false, false, 6>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n6_causal_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, false, true, 6>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n6_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, true, false, 6>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n6_causal_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, true, true, 6>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n12_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, false, false, 12>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n12_causal_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, false, true, 12>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n12_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, true, false, 12>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n12_causal_f32_d64")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<64, true, true, 12>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n12_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, false, false, 12>;
template [[host_name("kernel_flash_attn_train_back_dv_mm_n12_causal_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, false, true, 12>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n12_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, true, false, 12>;
template [[host_name("kernel_flash_attn_train_back_dk_mm_n12_causal_f32_d128")]] kernel kernel_flash_attn_train_back_kv_mm_f32_t kernel_flash_attn_train_back_kv_mm_f32<128, true, true, 12>;

// HOT-Step: vectorised f32 -> f32 copy for same-shape tensors with 16-byte-aligned contiguous rows
// (CPY/CONT of views, and the src0 -> dst copy of non-inplace ACC/SET). One float4 per thread.
// Grid: (nw0*ne01, ne02, ne03), threads (nth, 1, 1), nw0 = ceil(ne00/4 / nth). Selected by the host
// (hot_step_cpy_v4_ok); GGML_METAL_CPY_V4=0 falls back to kernel_cpy_f32_f32.
kernel void kernel_cpy_f32_f32_v4(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = tgpig[0]%args.ne01;
    const int32_t iw0 = tgpig[0]/args.ne01;
    const int32_t i   = iw0*ntg[0] + tpitg.x;
    if (i >= args.ne00/4) {
        return;
    }
    *(device float4 *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1 + i*16) =
        *(device const float4 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i*16);
}

// HOT-Step (B9): f32 mul_mat for a reduction length of exactly 32 (the LoRA
// rank), kernel_mul_mm_k32_f32. ggml's kernel_mul_mm needs ne00 >= 64 and, on
// Apple7 without the tensor API, rounds its F32 operands to half; both
// reasons send these matmuls (dst = [M, N], M up to 12288, N ~ 9.5k-15.5k,
// K = 32) to kernel_mul_mv_f32_f32, which runs one 32-thread reduction per
// output element-row and was ~11 % of the AR backward GPU time. This kernel is
// a plain tiled GEMM with genuine float32 operands and accumulation
// (simdgroup_float8x8): dst[m, n] = sum_k src0[k, m] * src1[k, n].
// Not bit-identical to the matrix-vector kernel (blocked summation order);
// gated by yue2-mul-mat-k32-metal-test.cpp on float32 tolerance.
//
// Tile 64 (M) x 64 (N), 4 simdgroups (2x2, each 32x32 = 4x4 fragments), the whole
// K = 32 in one pass. The result is staged through threadgroup memory
// (reusing the operand tiles) so the device writes are full 256-byte rows.
// smem: sA 64*36 + sB 64*36 floats (the scratch 64*68 fits inside).
#define MK_NR0 64
#define MK_NR1 64
#define MK_LDS 36
#define MK_LDC 68

kernel void kernel_mul_mm_k32_f32(
        constant ggml_metal_kargs_mul_mm_k32 & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    // Occupancy rework (Weg 2 follow-up): K = 32 in two halves of 16 (smem 2*64*20 floats = 10 KB
    // instead of 18 KB) and the output staged in two 32-row halves through the same buffer.
    // Same k order per accumulator (0,8,16,24) -> bit-identical to the single-pass version.
    constexpr short KLDS = 20;
    threadgroup float * sA = (threadgroup float *) shmem;                       // [64 m][20]
    threadgroup float * sB = sA + MK_NR0*KLDS;                                   // [64 n][20]
    threadgroup float * sc = (threadgroup float *) shmem;                       // [32 n][68], reuses sA/sB

    const int n0 = int(tgpig.x)*MK_NR1;
    const int m0 = int(tgpig.y)*MK_NR0;

    const short mbase = (sgitg & 1)*32;
    const short nbase = (sgitg >> 1)*32;

    simdgroup_float8x8 mc[16];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 16; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (short kh = 0; kh < 2; ++kh) {
        if (kh > 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);   // previous half fully consumed
        }
        // stage 64 rows x 4 float4 (16 floats) per tile: 2 float4 per thread per tile
        for (short e = 0; e < 2; ++e) {
            const short idx = tiitg + 128*e;
            const short row = idx >> 2;
            const short c4  = idx & 3;

            const float4 a = *(device const float4 *) (src0 + (int64_t) (m0 + row)*args.nb01 + (kh*4 + c4)*16);
            *(threadgroup float4 *) (sA + row*KLDS + c4*4) = a;

            float4 b = float4(0.0f);
            if (n0 + row < args.N) {
                b = *(device const float4 *) (src1 + (int64_t) (n0 + row)*args.nb11 + (kh*4 + c4)*16);
            }
            *(threadgroup float4 *) (sB + row*KLDS + c4*4) = b;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (short kk = 0; kk < 16; kk += 8) {
            simdgroup_float8x8 ma[4];
            simdgroup_float8x8 mb[4];
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], sA + (mbase + 8*i)*KLDS + kk, KLDS);
                simdgroup_load(mb[i], sB + (nbase + 8*i)*KLDS + kk, KLDS, 0, true);
            }
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                #pragma clang loop unroll(full)
                for (short j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(mc[i*4 + j], ma[i], mb[j], mc[i*4 + j]);
                }
            }
        }
    }

    // two output phases: simdgroups with nbase == 0 first, then nbase == 32 (32 n-rows of scratch each)
    for (short ph = 0; ph < 2; ++ph) {
        threadgroup_barrier(mem_flags::mem_threadgroup);   // sc overlaps sA/sB (ph 0) / previous phase's reads (ph 1)
        if ((nbase == 0) == (ph == 0)) {
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                #pragma clang loop unroll(full)
                for (short j = 0; j < 4; ++j) {
                    // fragment is [m][n]; dst wants n rows with m contiguous -> transposed store
                    simdgroup_store(mc[i*4 + j], sc + (8*j)*MK_LDC + (mbase + 8*i), MK_LDC, 0, true);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short idx = tiitg; idx < 32*(MK_NR0/4); idx += 128) {
            const short row = idx >> 4;     // n offset within this half
            const short c4  = idx & 15;     // float4 along m
            const int   nr  = n0 + ph*32 + row;
            if (nr < args.N) {
                *(device float4 *) (dst + (int64_t) nr*args.nb1 + (int64_t) (m0 + c4*4)*4) =
                    *(threadgroup float4 *) (sc + row*MK_LDC + c4*4);
            }
        }
    }
}
