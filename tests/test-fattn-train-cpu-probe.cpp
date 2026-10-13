// S2c regression: the CPU backend must refuse BF16 Q/K/V for
// GGML_OP_FLASH_ATTN_TRAIN{,_BACK} and still accept F32. ggml.c's
// constructors widened to accept BF16 (same type across Q/K/V), but the CPU
// kernel in ops.cpp is F32-only and GGML_ABORTs on anything else. Without
// this check, ggml_backend_supports_op() on the CPU device would claim
// support for an op it cannot run.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <stdio.h>

static struct ggml_tensor * make_qkv(struct ggml_context * ctx, ggml_type type) {
    const int64_t D = 16, S = 4, H = 2, B = 1;
    return ggml_new_tensor_4d(ctx, type, D, S, H, B);
}

static bool supports(ggml_backend_t backend, ggml_type type, bool back) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 + 1 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * q = make_qkv(ctx, type);
    struct ggml_tensor * k = make_qkv(ctx, type);
    struct ggml_tensor * v = make_qkv(ctx, type);

    struct ggml_tensor * op;
    if (!back) {
        op = ggml_flash_attn_train(ctx, q, k, v, NULL, 1.0f);
    } else {
        struct ggml_tensor * fwd  = ggml_flash_attn_train(ctx, q, k, v, NULL, 1.0f);
        struct ggml_tensor * dfwd = ggml_dup(ctx, fwd);
        op = ggml_flash_attn_train_back(ctx, q, k, v, NULL, fwd, dfwd, 1.0f);
    }

    const bool ok = ggml_backend_supports_op(backend, op);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend && "ggml_backend_cpu_init() failed");

    GGML_ASSERT(supports(backend, GGML_TYPE_F32, /*back=*/false));
    GGML_ASSERT(supports(backend, GGML_TYPE_F32, /*back=*/true));
    GGML_ASSERT(!supports(backend, GGML_TYPE_BF16, /*back=*/false));
    GGML_ASSERT(!supports(backend, GGML_TYPE_BF16, /*back=*/true));

    ggml_backend_free(backend);
    printf("test-fattn-train-cpu-probe: OK\n");
    return 0;
}
