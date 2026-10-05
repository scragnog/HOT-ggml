#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

// HOT-Step patch: quant-cpy-generic - see engine/patches/quant-cpy-kquant.patch
// True when ggml_cuda_cpy() can dequantize this type to F32 through
// ggml_get_to_fp32_cuda(). Exported so supports_op() and the dispatch answer the
// same question from one place instead of from two lists that drift apart.
bool ggml_cuda_cpy_quant_to_f32_supported(ggml_type type);

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
