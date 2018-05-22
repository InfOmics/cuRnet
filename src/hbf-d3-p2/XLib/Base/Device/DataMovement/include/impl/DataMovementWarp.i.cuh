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
#pragma once
#include "../../../Util/Util.cuh"
using namespace PTX;

namespace data_movement {
namespace warp {

enum MEM_SPACE { GLOBAL, SHARED };

/*
* not documented
*/
template<MEM_SPACE _MEM_SPACE, int SIZE, typename T>
void __device__ __forceinline__ WarpGlobalOffset(T* __restrict__ &Pointer) {
    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 : 1;

    if (_MEM_SPACE == GLOBAL && SIZE % AGGR_SIZE_16 == 0)
        Pointer += LaneID() * AGGR_SIZE_16;
    else if (SIZE % AGGR_SIZE_8 == 0)
        Pointer += LaneID() * AGGR_SIZE_8;
    else if (SIZE % AGGR_SIZE_4 == 0)
        Pointer += LaneID() * AGGR_SIZE_4;
    else if (SIZE % AGGR_SIZE_2 == 0)
        Pointer += LaneID() * AGGR_SIZE_2;
    else
        Pointer += LaneID();
}

//==============================================================================

template<cub::CacheStoreModifier M, int SIZE, typename T>
void __device__ __forceinline__ SharedToGlobalSupport(T* __restrict__ SMem,
                                                      T* __restrict__ Pointer) {

    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_16 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_16; i += WARP_SIZE)
            cub::ThreadStore<M>(reinterpret_cast<int4*>(Pointer) + i,
                                reinterpret_cast<int4*>(SMem)[i]);
    }
    else if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i += WARP_SIZE)
            cub::ThreadStore<M>(reinterpret_cast<int2*>(Pointer) + i,
                                reinterpret_cast<int2*>(SMem)[i]);
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i += WARP_SIZE)
            cub::ThreadStore<M>(reinterpret_cast<int*>(Pointer) + i,
                                reinterpret_cast<int*>(SMem)[i]);
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
       #pragma unroll
       for (int i = 0; i < SIZE / AGGR_SIZE_2; i += WARP_SIZE)
           cub::ThreadStore<M>(reinterpret_cast<short*>(Pointer) + i,
                               reinterpret_cast<short*>(SMem)[i]);
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i += WARP_SIZE)
            cub::ThreadStore<M>(Pointer + i, SMem[i]);
    }
}

template<cub::CacheLoadModifier M, int SIZE, typename T>
void __device__ __forceinline__ GlobalToSharedSupport(T* __restrict__ SMem,
                                                      T* __restrict__ Pointer) {

    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_16 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_16; i += WARP_SIZE)
            reinterpret_cast<int4*>(SMem)[i] =
                        cub::ThreadLoad<M>(reinterpret_cast<int4*>(Pointer) + i);
    }
    else if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i += WARP_SIZE)
            reinterpret_cast<int2*>(SMem)[i] =
                        cub::ThreadLoad<M>(reinterpret_cast<int2*>(Pointer) + i);
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i += WARP_SIZE)
            reinterpret_cast<int*>(SMem)[i] =
                        cub::ThreadLoad<M>(reinterpret_cast<int*>(Pointer) + i);
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
       #pragma unroll
       for (int i = 0; i < SIZE / AGGR_SIZE_2; i += WARP_SIZE)
           reinterpret_cast<short*>(SMem)[i] =
                       cub::ThreadLoad<M>(reinterpret_cast<short*>(Pointer) + i);
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i += WARP_SIZE)
            SMem[i] = cub::ThreadLoad<M>(Pointer + i);
    }
}

//==============================================================================

template<int SIZE, cub::CacheStoreModifier M, typename T>
void __device__ __forceinline__ SharedToGlobal(T* __restrict__ SMem,
                                               T* __restrict__ Pointer) {
    WarpGlobalOffset<GLOBAL, SIZE>(SMem);
    WarpGlobalOffset<GLOBAL, SIZE>(Pointer);
    SharedToGlobalSupport<M, SIZE>(SMem, Pointer);
}

template<int SIZE, cub::CacheLoadModifier M, typename T>
void __device__ __forceinline__ GlobalToShared(T* __restrict__ Pointer,
                                               T* __restrict__ SMem) {
    WarpGlobalOffset<GLOBAL, SIZE>(SMem);
    WarpGlobalOffset<GLOBAL, SIZE>(Pointer);
    GlobalToSharedSupport<M, SIZE>(Pointer, SMem);
}

} //@warp
} //@data_movement
