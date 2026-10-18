#include "common.h"

// Load along the source's contiguous dimension, then store along the destination's.
// Integer storage preserves every F32/BF16 bit, including NaN payloads and signed zero.
template<typename T>
kernel void kernel_cpy_transpose(
        constant ggml_metal_kargs_cpy & args [[buffer(0)]],
        device const char * src [[buffer(1)]],
        device char * dst [[buffer(2)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 lane [[thread_position_in_threadgroup]]) {
    threadgroup T tile[32][33];
    const ulong i2 = group.z % args.ne02;
    const ulong i3 = group.z / args.ne02;
    src += i2*args.nb02 + i3*args.nb03;
    dst += i2*args.nb2  + i3*args.nb3;
    const ulong x = ulong(group.x)*32;
    const ulong y = ulong(group.y)*32;

    for (uint j = lane.y; j < 32; j += 8) {
        if (x + j < ulong(args.ne00) && y + lane.x < ulong(args.ne01)) {
            tile[j][lane.x] = *(device const T *)(src + (x+j)*args.nb00 + (y+lane.x)*args.nb01);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = lane.y; j < 32; j += 8) {
        if (x + lane.x < ulong(args.ne00) && y + j < ulong(args.ne01)) {
            *(device T *)(dst + (x+lane.x)*args.nb0 + (y+j)*args.nb1) = tile[lane.x][j];
        }
    }
}

template [[host_name("kernel_cpy_transpose_f32")]]
kernel void kernel_cpy_transpose<uint>(constant ggml_metal_kargs_cpy &, device const char *, device char *, uint3, uint3);
template [[host_name("kernel_cpy_transpose_bf16")]]
kernel void kernel_cpy_transpose<ushort>(constant ggml_metal_kargs_cpy &, device const char *, device char *, uint3, uint3);
