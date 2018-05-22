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
#include "DataMovementWarp.i.cuh"
#include "../../../Util/Util.cuh"
#include "../../../../Host/BaseHost.hpp"

namespace data_movement {
namespace warp {
namespace ordered {

using namespace PTX;
using namespace numeric;

namespace {

/**
* SMem must be in the correct position for each lane
*/
template<int SIZE, typename T>
void __device__ __forceinline__ SharedRegSupport(T* __restrict__ Source,
                                                 T* __restrict__ Dest) {
    if (sizeof(T) > 8) {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            Dest[i] = Source[i];
    }
    else {
        const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
        const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
        const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            reinterpret_cast<int2*>(Dest)[i] = reinterpret_cast<int2*>(Source)[i];

        const int VISITED_8 = (SIZE / AGGR_SIZE_8) * AGGR_SIZE_8;
        const int START_4 = VISITED_8 / AGGR_SIZE_4;
        if (START_4 < SIZE / AGGR_SIZE_4)
            reinterpret_cast<int*>(Dest)[START_4] = reinterpret_cast<int*>(Source)[START_4];

        const int VISITED_4 = VISITED_8 + (SIZE % AGGR_SIZE_8) / AGGR_SIZE_4;
        const int START_2 =  VISITED_4 / AGGR_SIZE_2;
        if (START_2 < SIZE / AGGR_SIZE_2)
            reinterpret_cast<short*>(Dest)[START_2] = reinterpret_cast<short*>(Source)[START_2];

        const int START_1 = VISITED_8 + VISITED_4 + (SIZE % AGGR_SIZE_8) / AGGR_SIZE_2;
        if (START_1 < SIZE)
            reinterpret_cast<char*>(Dest)[START_1] = reinterpret_cast<char*>(Source)[START_1];
    }
    /*    if (SIZE % AGGR_SIZE_8 != 2)
            reinterpret_cast<int*>(Dest)[SIZE / AGGR_SIZE_8] = reinterpret_cast<int*>(Source)[SIZE / AGGR_SIZE_8];
        if (SIZE % AGGR_SIZE_8 != 1)
            reinterpret_cast<short*>(Dest)[SIZE / AGGR_SIZE_4] = reinterpret_cast<short*>(Source)[SIZE / AGGR_SIZE_4];
    }

    if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            reinterpret_cast<int2*>(Dest)[i] = reinterpret_cast<int2*>(Source)[i];
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            reinterpret_cast<int*>(Dest)[i] = reinterpret_cast<int*>(Source)[i];
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
       #pragma unroll
       for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
           reinterpret_cast<short*>(Dest)[i] = reinterpret_cast<short*>(Source)[i];
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            Dest[i] = Source[i];
    }*/
}

//------------------------------------------------------------------------------

template<cub::CacheLoadModifier M, int RECURSIONS, int STEP = 1>
struct GlobalToRegSupport {

    template<int Items_per_warp, typename T, int SIZE>
    static __device__ __forceinline__ void Apply(T* __restrict__ Pointer,
                                                 T* __restrict__ SMemThread,
                                                 int& offset,
                                                 int& j,
                                                 T* __restrict__ SMem,
                                                 T (&Queue)[SIZE]) {

        while (j < SIZE && offset < Items_per_warp) {
			Queue[j] = SMemThread[offset];
			j++;
            offset++;
		}
        const int SHARED_ITEMS = MIN<Items_per_warp, SIZE - (STEP - 1) * Items_per_warp>::value;
        warp::GlobalToSharedSupport<M, SHARED_ITEMS>(Pointer, SMem);
        offset -= Items_per_warp;
        Pointer += Items_per_warp;

        GlobalToRegSupport<M, RECURSIONS, STEP + 1>::
             Apply<Items_per_warp>(Pointer, SMemThread, offset, j, SMem, Queue);
    }
};

template<cub::CacheLoadModifier M, int RECURSIONS>
struct GlobalToRegSupport<M, RECURSIONS, RECURSIONS> {
    template<int Items_per_warp, typename T, int SIZE>
    static __device__ __forceinline__ void Apply(T* __restrict__ Pointer,
                                                 T* __restrict__ SMemThread,
                                                 int& offset,
                                                 int& j,
                                                 T* __restrict__ SMem,
                                                 T (&Queue)[SIZE]) {}
};

//------------------------------------------------------------------------------

template<cub::CacheStoreModifier M, int RECURSIONS, int STEP = 1>
struct RegToGlobalSupport {

    template<int Items_per_warp, typename T, int SIZE>
    static __device__ __forceinline__ void Apply(T (&Queue)[SIZE],
                                                 T* __restrict__ SMemThread,
                                                 int& offset,
                                                 int& j,
                                                 T* __restrict__ SMem,
                                                 T* __restrict__ Pointer) {

        while (j < SIZE && offset < Items_per_warp) {
			SMemThread[offset] = Queue[j];
			j++;
            offset++;
		}
        const int SHARED_ITEMS = MIN<Items_per_warp, SIZE - (STEP - 1) * Items_per_warp>::value;
        warp::SharedToGlobalSupport<M, SHARED_ITEMS>(SMem, Pointer);
        offset -= Items_per_warp;
        Pointer += Items_per_warp;

        RegToGlobalSupport<M, RECURSIONS, STEP + 1>::
            Apply<Items_per_warp>(Queue, SMemThread, offset, j, SMem, Pointer);
    }
};

template<cub::CacheStoreModifier M, int RECURSIONS>
struct RegToGlobalSupport<M, RECURSIONS, RECURSIONS> {
    template<int Items_per_warp, typename T, int SIZE>
    static __device__ __forceinline__ void Apply(T (&Queue)[SIZE],
                                                 T* __restrict__ SMemThread,
                                                 int& offset,
                                                 int& j,
                                                 T* __restrict__ SMem,
                                                 T* __restrict__ Pointer) {}
};

} //@anonymous

//==============================================================================


template<cub::CacheLoadModifier M, typename T,
         int Items_per_warp, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {

    if (SIZE * WARP_SIZE <= Items_per_warp) {
        T* SMemThread = SMem + LaneID() * SIZE;     // OK

        warp::WarpGlobalOffset<GLOBAL, SIZE * WARP_SIZE>(SMem);
        warp::WarpGlobalOffset<GLOBAL, SIZE * WARP_SIZE>(Pointer);
        warp::GlobalToSharedSupport<M, SIZE * WARP_SIZE>(Pointer, SMem);

        if (threadIdx.x == 0) {
            for (int i = 0; i < 6 * 32; i++) {
                printf("%d ", SMemThread[i]);
            }
            printf("\n");
        }

        SharedRegSupport<SIZE>(SMemThread, Queue);
    } else {
        T* SMemThread = SMem;

        warp::WarpGlobalOffset<GLOBAL, SIZE>(SMem);
        warp::WarpGlobalOffset<GLOBAL, SIZE>(Pointer);

        int offset = LaneID() * SIZE, j = 0;
        const int RECURSIONS = _Div(SIZE * WARP_SIZE, Items_per_warp);
        GlobalToRegSupport<M, RECURSIONS>::
            Apply<Items_per_warp>(Pointer, SMemThread, offset, j, SMem, Queue);
    }
}

template<cub::CacheStoreModifier M, typename T, int Items_per_warp, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ SMem,
                                            T* __restrict__ Pointer) {

    if (SIZE * WARP_SIZE <= Items_per_warp) {
        T* SMemThread = SMem + LaneID() * SIZE;     // OK
        SharedRegSupport<SIZE>(Queue, SMemThread);

        warp::WarpGlobalOffset<GLOBAL, SIZE * WARP_SIZE>(SMem);
        warp::WarpGlobalOffset<GLOBAL, SIZE * WARP_SIZE>(Pointer);

        warp::SharedToGlobalSupport<M, SIZE * WARP_SIZE>(SMem, Pointer);
    } else {
        T* SMemThread = SMem;

        warp::WarpGlobalOffset<GLOBAL, SIZE>(SMem);
        warp::WarpGlobalOffset<GLOBAL, SIZE>(Pointer);

        int offset = LaneID() * SIZE, j = 0;
        const int RECURSIONS = _Div(SIZE * WARP_SIZE, Items_per_warp);
        RegToGlobalSupport<M, RECURSIONS>::
            Apply<Items_per_warp>(Queue, SMemThread, offset, j, SMem, Pointer);
    }
}

template<typename T, int SIZE>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem) {
    SMem += LaneID() * SIZE;
    SharedRegSupport<SIZE>(Queue, SMem);
}

template<typename T, int SIZE>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    SMem += LaneID() * SIZE;
    SharedRegSupport<SIZE>(SMem, Queue);
}

} //@ordered
} //@warp
} //@data_movement
