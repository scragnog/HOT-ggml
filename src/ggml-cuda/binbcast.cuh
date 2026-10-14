#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

// BF16_ROUND epilogue on an F32 ADD/SUB/MUL: dst is the round node.
void ggml_cuda_op_bin_bf16_round(ggml_backend_cuda_context & ctx, const ggml_tensor * bin, ggml_tensor * dst);
// The same op that also rounds src0 and/or src1 as it loads them (their
// deferred BF16_ROUND); dst is bin, or its round when round_dst.
void ggml_cuda_op_bin_bf16_round_inputs(ggml_backend_cuda_context & ctx, const ggml_tensor * bin, ggml_tensor * dst,
                                        bool round_src0, bool round_src1, bool round_dst);
