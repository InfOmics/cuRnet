#pragma once

#include <cub/cub.cuh>
// ----------- THREAD PTX -------------------

namespace scc4k{

__device__ __forceinline__ int LaneID() {
    int ret;
    asm("mov.u32 %0, %laneid;" : "=r"(ret) );
    return ret;
}

__device__ __forceinline__ int WarpID() {
	return Tid >> 5;
}

__device__ __forceinline__ void ThreadExit() {
	asm("exit;");
}


// ----------- MATH -------------------

// Three-operand add
__device__ __forceinline__ unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z) {
	asm("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(x) : "r"(x), "r"(y), "r"(z));
	return x;
}

// brief Shift-right then add.  Returns (x >> shift) + addend.
__device__ __forceinline__ unsigned int SHR_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend) {

    unsigned int ret;
    asm("vshr.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
        "=r"(ret) : "r"(x), "r"(shift), "r"(addend));
    return ret;
}


// brief Shift-left then add.  Returns (x << shift) + addend.
__device__ __forceinline__ unsigned int SHL_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend) {

    unsigned int ret;
    asm("vshl.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
        "=r"(ret) : "r"(x), "r"(shift), "r"(addend));
    return ret;
}

// ----------- LANE MASK -------------------


__device__ __forceinline__ unsigned int LaneMaskEq() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_eq;" : "=r"(ret) );
	return ret;
}

__device__ __forceinline__ unsigned int LaneMaskLt() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret) );
	return ret;
}

__device__ __forceinline__ unsigned int LaneMaskLe() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_le;" : "=r"(ret) );
	return ret;
}

__device__ __forceinline__ unsigned int LaneMaskGt() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_gt;" : "=r"(ret) );
	return ret;
}

__device__ __forceinline__ unsigned int LaneMaskGe() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_ge;" : "=r"(ret) );
	return ret;
}

}
