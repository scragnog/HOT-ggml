// HOT-Step patch: flash-attn-train (Vulkan). Shared by the three kernels.
//
// One 32-lane cluster per row (a query row for the forward and dQ, a key row
// for dK/dV), four rows per workgroup. Lane l holds elements l, l+32, ... of
// the D-wide vectors, so a dot product is a per-lane partial followed by a
// shuffle-xor reduction that never leaves the cluster. Every branch taken
// inside the row loop depends only on the row, so a cluster never diverges.
//
// Layouts match the CPU oracle (ggml-cpu/ops.cpp):
//   q [D, S, Nh, B], k/v [D, S_kv, Nkv, B], strides in floats;
//   mask [S_kv, S, m2, m3] f16, broadcast by modulo;
//   packed forward: O [D, Nh, S, B] at 0, LSE [Nh, S, B] at lse_off;
//   packed backward: dQ [D, S, Nh, B] at 0, dK at dk_off, dV at dv_off.

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require

#define FA_MAXE 8   // D <= 256

layout(local_size_x = 32, local_size_y = 4, local_size_z = 1) in;

layout(push_constant) uniform parameter {
    uint D, S, Nh, B, S_kv, Nkv;
    uint q_nb1, q_nb2, q_nb3;
    uint k_nb1, k_nb2, k_nb3;
    uint v_nb1, v_nb2, v_nb3;
    uint m_ne0, m_ne1, m_ne2, m_ne3, has_mask;
    float scale;
    uint lse_off, dk_off, dv_off;
    uint q_off, k_off, v_off, m_off, f_off, g_off, d_off;  // misalignment, in elements
    uint row_base;  // first row of this dispatch: the host slices long ones
} p;

layout(binding = 0) readonly buffer Qb { float q_d[]; };
layout(binding = 1) readonly buffer Kb { float k_d[]; };
layout(binding = 2) readonly buffer Vb { float v_d[]; };
layout(binding = 3) readonly buffer Mb { uint m_d[]; };   // f16 pairs

float fa_reduce(float x) {
    [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
        x += subgroupShuffleXor(x, s);
    }
    return x;
}

// Additive mask value for (head h, batch b, query i, key j); 0 without a mask.
float fa_mask(uint h, uint b, uint i, uint j) {
    if (p.has_mask == 0) {
        return 0.0;
    }
    const uint idx = p.m_off + j + p.m_ne0*(i + p.m_ne1*((h % p.m_ne2) + p.m_ne2*(b % p.m_ne3)));
    const vec2 pair = unpackHalf2x16(m_d[idx >> 1]);
    return (idx & 1u) == 0u ? pair.x : pair.y;
}

bool fa_is_neg_inf(float x) {
    return isinf(x) && x < 0.0;
}
