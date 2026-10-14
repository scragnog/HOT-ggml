#include "out-prod.cuh"
#include "convert.cuh" // HOT-Step patch: BF16 out_prod — see engine/patches/bf16-out-prod.patch

#include <cstdint>

static __global__ void k_compute_out_prod_ptrs(
        const float * src0_d, const float * src1_d, float * dst_d,
        const float ** ptrs_a, const float ** ptrs_b, float ** ptrs_c,
        const int64_t ne2, const int64_t ne3,
        const int64_t dps2, const int64_t dps3,
        const size_t s02, const size_t s03,
        const size_t s12, const size_t s13,
        const size_t s2,  const size_t s3) {
    const int64_t i2 = blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i3 = blockIdx.y*blockDim.y + threadIdx.y;

    if (i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t idx = i3*ne2 + i2;

    ptrs_a[idx] = src0_d + (i3/dps3)*s03 + (i2/dps2)*s02;
    ptrs_b[idx] = src1_d +  i3      *s13 +  i2      *s12;
    ptrs_c[idx] = dst_d  +  i3      *s3  +  i2      *s2;
}

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // HOT-Step patch: BF16 out_prod — see engine/patches/bf16-out-prod.patch
    // ggml computes the gradient w.r.t. a mul_mat's ACTIVATION input as
    // out_prod(weight, transpose(grad)), so the frozen weight lands in src0, and
    // BF16-stored activations land in src1. Either may be BF16; dst stays F32.
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    // HOT-Step patch: BF16 out_prod — see engine/patches/bf16-out-prod.patch
    // A BF16 source is widened (exactly) into an F32 workspace that mirrors its
    // element layout: the whole byte span [data, data + nbytes) is converted
    // flat, so every element stride is unchanged and only the element size
    // differs. The GEMMs below are then the shipped F32 ones with the same
    // lda/ldb/batch strides an F32 copy of the input would have had, so the
    // result is bit-identical to running on F32 copies.
    const auto widen = [&](const ggml_tensor * t, ggml_cuda_pool_alloc<float> & buf) -> const float * {
        if (t->type == GGML_TYPE_F32) {
            return (const float *) t->data;
        }
        const int64_t n = ggml_nbytes(t) / ggml_type_size(t->type);
        buf.alloc(n);
        ggml_get_to_fp32_cuda(GGML_TYPE_BF16)(t->data, buf.get(), n, stream);
        return buf.get();
    };
    ggml_cuda_pool_alloc<float> src0_f32(ctx.pool());
    ggml_cuda_pool_alloc<float> src1_f32(ctx.pool());

    const float * src0_d = widen(src0, src0_f32);
    const float * src1_d = widen(src1, src1_f32);
    float       *  dst_d = (float       *)  dst->data;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // element sizes: strides in elements are the same in the F32 workspace
    const size_t ts0 = ggml_type_size(src0->type);
    const size_t ts1 = ggml_type_size(src1->type);

    const int64_t lda = nb01 / ts0;
    const int64_t ldc = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op =  src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t           ldb            = (src1_T ?        nb10 :        nb11) /  ts1;
    GGML_ASSERT(                             (src1_T ?        nb11 :        nb10) == ts1);

    // data strides in dimensions 2/3
    const size_t s02 = nb02 / ts0;
    const size_t s03 = nb03 / ts0;
    const size_t s12 = nb12 / ts1;
    const size_t s13 = nb13 / ts1;
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    if (dps2 == 1 && ne2 > 1) {
        // src0 has uniform stride s02 along dim 2; batch the inner loop with a strided GEMM
        GGML_ASSERT(ne2 <= std::numeric_limits<int>::max());
        const int batch_count = (int) ne2;
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            CUBLAS_CHECK(
                cublasSgemmStridedBatched(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03, lda, s02,
                                src1_d +  i3     *s13, ldb, s12,
                        &beta,  dst_d  +  i3     *s3,  ldc, s2,
                        batch_count));
        }
    } else if (ne2 > 1 || ne3 > 1) {
        // dps2 > 1 (src0 broadcast along dim 2 with non-uniform stride) or multiple GEMMs
        // along dim 3: compute per-GEMM pointers on the device and use a single batched GEMM.
        GGML_ASSERT(ne3 > 0);
        GGML_ASSERT(ne2 <= (int64_t) std::numeric_limits<int>::max() / ne3);
        const int batch_count = (int) (ne2 * ne3);

        ggml_cuda_pool_alloc<const float *> ptrs_a(ctx.pool(), batch_count);
        ggml_cuda_pool_alloc<const float *> ptrs_b(ctx.pool(), batch_count);
        ggml_cuda_pool_alloc<      float *> ptrs_c(ctx.pool(), batch_count);

        const dim3 block_dims(16, 16);
        const dim3 grid_dims((ne2 + block_dims.x - 1)/block_dims.x, (ne3 + block_dims.y - 1)/block_dims.y);
        k_compute_out_prod_ptrs<<<grid_dims, block_dims, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ptrs_a.get(), ptrs_b.get(), ptrs_c.get(),
            ne2, ne3, dps2, dps3, s02, s03, s12, s13, s2, s3);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
            cublasSgemmBatched(handle, CUBLAS_OP_N, src1_cublas_op,
                    ne0, ne1, ne01,
                    &alpha, ptrs_a.get(), lda,
                            ptrs_b.get(), ldb,
                    &beta,  ptrs_c.get(), ldc,
                    batch_count));
    } else {
        // ne2 == 1 && ne3 == 1: single GEMM
        CUBLAS_CHECK(
            cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                    ne0, ne1, ne01,
                    &alpha, src0_d, lda,
                            src1_d, ldb,
                    &beta,  dst_d,  ldc));
    }
}
