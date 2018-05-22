/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

H-BF is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include "../../../../Host/BaseHost.hpp"
using namespace numeric;

namespace PTX {

__device__ __forceinline__ unsigned int LaneID() {
    unsigned int ret;
    asm ("mov.u32 %0, %laneid;" : "=r"(ret) );
    return ret;
}

__device__ __forceinline__ void ThreadExit() {
	asm ("exit;");
}

// Three-operand add    // MAXWELL
__device__ __forceinline__ unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z) {
	asm ("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(x) : "r"(x), "r"(y), "r"(z));
	return x;
}

__device__ __forceinline__ unsigned int __fms(unsigned int word) {
    unsigned int ret;
    asm ("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
}

__device__ __forceinline__ unsigned int __fms(unsigned long long dword) {
    unsigned int ret;
    asm ("bfind.u64 %0, %1;" : "=r"(ret) : "l"(dword));
    return ret;
}

__device__ __forceinline__ unsigned int __fms(int word) {
    unsigned int ret;
    asm ("bfind.s32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
}

__device__ __forceinline__ unsigned int __fms(long long dword) {
    unsigned int ret;
    asm ("bfind.s64 %0, %1;" : "=r"(ret) : "l"(dword));
    return ret;
}

    // brief Shift-right then add.  Returns (x >> shift) + addend.
    /*__device__ __forceinline__ unsigned int SHR_ADD(
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
    }*/
__device__ __forceinline__ unsigned int LaneMaskEQ() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_eq;" : "=r"(ret) );
	return ret;
}

__device__ __forceinline__ unsigned int LaneMaskLT() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret) );
	return ret;
}

__device__ __forceinline__ unsigned int LaneMaskLE() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_le;" : "=r"(ret) );
	return ret;
}

__device__ __forceinline__ unsigned int LaneMaskGT() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_gt;" : "=r"(ret) );
	return ret;
}

__device__ __forceinline__ unsigned int LaneMaskGE() {
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_ge;" : "=r"(ret) );
	return ret;
}

} //@PTX
