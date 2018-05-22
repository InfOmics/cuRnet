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
#include "../../../Util/Util.cuh"

namespace primitives {

using namespace numeric;
using namespace PTX;

namespace {

#define warpReduce(ASM_OP, ASM_T, ASM_CL)                                      \
_Pragma("unroll")														       \
for (int STEP = 0; STEP < LOG2<WARP_SZ>::value; STEP++) {				       \
    const int MASK_WARP = (1 << (STEP + 1)) - 1;						       \
    const int C = ((31 - MASK_WARP) << 8) | (MASK_WARP) ;				       \
    asm(																       \
        "{"																       \
        ".reg ."#ASM_T" r1;"											       \
        ".reg .pred p;"													       \
        "shfl.down.b32 r1|p, %1, %2, %3;"								       \
        "@p "#ASM_OP"."#ASM_T" r1, r1, %1;"								       \
        "mov."#ASM_T" %0, r1;"											       \
        "}"																       \
        : "="#ASM_CL(value) : #ASM_CL(value), "r"(1 << STEP), "r"(C));		   \
}																		       \
if (BROADCAST)															       \
    value = __shfl(value, 0, WARP_SZ);

//==============================================================================

template<int WARP_SZ, bool BROADCAST, typename T>
struct WarpReduceHelper {
    static __device__ __forceinline__ void Add(T& value);
    static __device__ __forceinline__ void Min(T& value);
    static __device__ __forceinline__ void Max(T& value);
};

template<int WARP_SZ, bool BROADCAST>
struct WarpReduceHelper<WARP_SZ, BROADCAST, int> {
    static __device__ __forceinline__ void Add(int& value) {
        warpReduce(add, s32, r)
    }
    static __device__ __forceinline__ void Min(int& value) {
        warpReduce(min, s32, r)
    }
    static __device__ __forceinline__ void Max(int& value) {
        warpReduce(max, s32, r)
    }
};

template<int WARP_SZ, bool BROADCAST>
struct WarpReduceHelper<WARP_SZ, BROADCAST, float> {
    static __device__ __forceinline__ void Add(float& value) {
        warpReduce(add, f32, f)
    }
    static __device__ __forceinline__ void Min(float& value) {
        warpReduce(min, f32, f)
    }
    static __device__ __forceinline__ void Max(float& value) {
        warpReduce(max, f32, f)
    }
};
} //@anonymous

//==============================================================================

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::Add(T& value) {
    WarpReduceHelper<WARP_SZ, false, T>::Add(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::Min(T& value) {
    WarpReduceHelper<WARP_SZ, false, T>::Min(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::Max(T& value) {
    WarpReduceHelper<WARP_SZ, false, T>::Max(value);
}

//------------------------------------------------------------------------------

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::AddBcast(T& value) {
    WarpReduceHelper<WARP_SZ, true, T>::Add(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::MinBcast(T& value) {
    WarpReduceHelper<WARP_SZ, true, T>::Min(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::MaxBcast(T& value) {
    WarpReduceHelper<WARP_SZ, true, T>::Max(value);
}

//------------------------------------------------------------------------------

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::Add(T& value, T* pointer) {
    WarpReduceHelper<WARP_SZ, false, T>::Add(value);
    if (LaneID() == 0)
        *pointer = value;
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::Min(T& value, T* pointer) {
    WarpReduceHelper<WARP_SZ, false, T>::Min(value);
    if (LaneID() == 0)
        *pointer = value;
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::Max(T& value, T* pointer) {
    WarpReduceHelper<WARP_SZ, false, T>::Max(value);
    if (LaneID() == 0)
        *pointer = value;
}

//------------------------------------------------------------------------------

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::AddAtom(T& value, T* pointer) {
    WarpReduceHelper<WARP_SZ, false, T>::Add(value);
    if (LaneID() == 0)
        atomicAdd(pointer, value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::MinAtom(T& value, T* pointer) {
    WarpReduceHelper<WARP_SZ, false, T>::Min(value);
    if (LaneID() == 0) {
        if (std::is_same<typename std::remove_cv<T>::type, unsigned int>::value ||
            std::is_same<typename std::remove_cv<T>::type, int>::value) {
            atomicMin(pointer, value);
        }
        else if (std::is_same<typename std::remove_cv<T>::type, float>::value)
            atomicMin(reinterpret_cast<unsigned int*>(pointer),
                      reinterpret_cast<unsigned int&>(value));
        else if (std::is_same<typename std::remove_cv<T>::type, double>::value)
            atomicMin(reinterpret_cast<long long unsigned int*>(pointer),
                      reinterpret_cast<long long unsigned int&>(value));
    }
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__ void WarpReduce<WARP_SZ>::MaxAtom(T& value, T* pointer) {
    WarpReduceHelper<WARP_SZ, false, T>::Max(value);
    if (LaneID() == 0) {
        if (std::is_same<typename std::remove_cv<T>::type, unsigned int>::value ||
            std::is_same<typename std::remove_cv<T>::type, int>::value) {
            atomicMax(pointer, value);
        }
        else if (std::is_same<typename std::remove_cv<T>::type, float>::value)
            atomicMax(reinterpret_cast<unsigned int*>(pointer),
                      reinterpret_cast<unsigned int&>(value));
        else if (std::is_same<typename std::remove_cv<T>::type, double>::value)
            atomicMax(reinterpret_cast<long long unsigned int*>(pointer),
                      reinterpret_cast<long long unsigned int&>(value));
    }
}
#undef warpReduce

} //@primitives
