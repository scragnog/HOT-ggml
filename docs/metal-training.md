# Metal backend: training ops

This fork's Metal backend implements the ops needed to train on Apple Silicon, not only to run inference. Everything lives in `src/ggml-metal/` (kernels in `kernels/hotstep_train.metal` and `ggml-metal.metal`, host side in `ggml-metal-ops.cpp`). Ops added in the core and CPU backend: `RMS_NORM_BACK`, `REPEAT_BACK`, `CONVROT8` / `CONVROT8_BACK`, `FLASH_ATTN_TRAIN` / `FLASH_ATTN_TRAIN_BACK`, `GGML_UNARY_OP_BF16_ROUND`, plus OUT_PROD and SILU_BACK kernels on Metal. Reference results come from the CPU backend; `tests/` and `tools` of the consuming project compare against it.

## Fused training attention (FLASH_ATTN_TRAIN)

`FLASH_ATTN_TRAIN` is a fused attention op with a packed O+LSE output and a hand-written backward, `FLASH_ATTN_TRAIN_BACK` (one output tensor holding dQ|dK|dV, followed by scratch regions the op owns). GQA, a causal hint, an optional mask tensor and a detached key prefix (`ggml_flash_attn_train_{set,get}_kv_grad_start`, op_params slot 4: keys below it get no K/V gradient) are supported.

Backward passes, in order: delta (`sum_d dO*O`), dV, dK, dQ.

**DSW (dS materialisation, default on).** The dK kernel also writes dS as 8x8 f32 tiles to a scratch region; a light second kernel computes `dQ += dS * K` from those tiles instead of recomputing S, P and dP. The per-element math and the accumulation order are identical to the recompute kernels, so the results are bit-identical (checked by `tests`/`fattn-train-test --dsw-check`). It is used when D is 64 or 128 and either the op is causal with `kv_grad_start == 0`, or it is non-causal without a mask tensor. In the non-causal case the old dQ kernel handles the detached key prefix and the DSW reader continues from its accumulator. Other shapes use the recompute kernels.

The scratch lives inside the op's destination buffer, after dQ|dK|dV and delta. `ggml_backend_metal_buffer_type_get_alloc_size` reserves it (`ggml_metal_op_flash_attn_train_back_extra_dsw`), the graph allocator can reuse that memory after the op, and the encoder and the size function share one plan (`hs_fa_dsw_plan_get`). The scratch is per group of kv heads (`GGML_METAL_FA_TRAIN_DSW_HKG`, default 2), not per op, which keeps it at a few GiB for long sequences.

## Environment variables

| Variable | Effect |
| --- | --- |
| `GGML_METAL_FA_TRAIN_DSW` | `0` selects the recompute backward kernels (default: DSW on). |
| `GGML_METAL_FA_TRAIN_DSW_HKG` | kv heads per scratch group (default 2); bigger means more scratch. |
| `GGML_METAL_FA_TRAIN_DSW_NSG`, `_DSW_NC` | dQ reader tile shape (defaults 8 and 16). |
| `GGML_METAL_FA_TRAIN_MM3_BWD`, `_MM3_BWD_KV` | `0` disables the simdgroup-matrix dQ and dK/dV kernels, which also turns DSW off. |
| `GGML_METAL_FA_TRAIN_DSW_SKIP`, `_DSW_VAR` | diagnostics only; `SKIP` gives wrong results by design. |
| `GGML_METAL_CR8B` | `0` selects the previous ConvRot8 backward kernel. |
| `GGML_METAL_CR8B_LMHEAD` | `1` also uses the half-precision ConvRot8 backward for very wide outputs; not bit-identical (off by default). |
| `GGML_METAL_CR8B_CHECK` | debug: run both ConvRot8 kernels and compare bitwise. |
| `GGML_METAL_OUT_PROD_TILED` | `0` selects the per-row OUT_PROD kernel instead of the tiled one. |
| `GGML_METAL_IM2COL_IC` | `0` selects the old im2col kernel. |

## Tests

`FLASH_ATTN_TRAIN` and the other training ops are exercised by the consuming project's `fattn-train-test` (`--backend metal`, `--dsw-check` for DSW bit-identity) and the `yue2-*` op tests. Output of the DSW check is `dq/dk/dv/packed diffs: 0/0/0/0  bitwise PASS` per case.
