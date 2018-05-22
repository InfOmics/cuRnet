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
#include "../../../Util/Util.cuh"
#include "../../../Primitives/Primitives.cuh"
using namespace PTX;
using namespace basic;
using namespace primitives;

namespace data_movement {
namespace dynamic {
namespace thread {

template<cub::CacheStoreModifier M, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            const int size,
                                            T* __restrict__ devPointer) {
    for (int i = 0; i < size; i++)
        cub::ThreadStore<M>(devPointer + i, Queue[i]);
}

template<cub::CacheStoreModifier M, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal_Unroll(T (&Queue)[SIZE],
                                            const int size,
                                            T* __restrict__ devPointer) {
    #pragma unroll
    for (int i = 0; i < SIZE; i++) {
        if (i < size)
            cub::ThreadStore<M>(devPointer + i, Queue[i]);
    }
}

} // @thread

namespace warp {

template<cub::CacheStoreModifier M, int Items_per_warp, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            const int size,
                                            int thread_mem_offset,
                                            const int total,
                                            T* __restrict__ SMem,
                                            T* __restrict__ devPointer) {

    T* SMemTMP = SMem;
    devPointer += LaneID();
    SMem += LaneID();
    int j = 0;
	while (true) {
		while (j < size && thread_mem_offset < Items_per_warp) {
			SMemTMP[thread_mem_offset] = Queue[j];
			j++;
            thread_mem_offset++;
		}
        if (total < Items_per_warp) {
            #pragma unroll
            for (int i = 0; i < Items_per_warp; i += 32) {
                if (LaneID() + i < total)
                    cub::ThreadStore<M>(devPointer + i, SMem[i]);
            }
            break;
        }
        else {
    		#pragma unroll
    		for (int i = 0; i < Items_per_warp; i += 32)
				cub::ThreadStore<M>(devPointer + i, SMem[i]);
        }
        total -= Items_per_warp;
		thread_mem_offset -= Items_per_warp;
		devPointer += Items_per_warp;
	}
}


template<cub::CacheStoreModifier M, int Items_per_warp, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal_Min(T (&Queue)[SIZE],
                                                const int size,
                                                int thread_mem_offset,
                                                const int total,
                                                T* __restrict__ SMem,
                                                T* __restrict__ devPointer) {
    int minValue = size;
	WarpReduce<>::MinBcast(minValue);

    T* devPointerTMP = devPointer + LaneID();
    for (int i = 0; i < minValue; i++)
        cub::ThreadStore<M>(devPointerTMP + i * WARP_SIZE, Queue[i]);

    size -= minValue;
	thread_mem_offset -= LaneID() * minValue;
    total -= minValue * WARP_SIZE;
    devPointer += minValue * WARP_SIZE;

    RegToGlobal(Queue + minValue, size, thread_mem_offset, total, SMem, devPointer);
}


template<cub::CacheStoreModifier M, int Items_per_warp, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal_Unroll(T (&Queue)[SIZE],
                                            const int size,
                                            int thread_mem_offset,
                                            const int total,
                                            T* __restrict__ SMem,
                                            T* __restrict__ devPointer) {
    T* SMemTMP = SMem;
    devPointer += LaneID();
    SMem += LaneID();
	while (true) {
        const int sum = thread_mem_offset + size;
        if (sum < Items_per_warp) {
            T* SMemTMP2 = SMemTMP + thread_mem_offset;
            #pragma unroll
            for (int i = 0; i < SIZE; i++) {
                if (i < size)
                    SMemTMP2[i] = Queue[i];
            }
        }
        const int partial = WarpBroadcast(thread_mem_offset,
                                          thread_mem_offset < Items_per_warp &&
                                          sum >= Items_per_warp);
        #pragma unroll
        for (int i = 0; i < Items_per_warp; i += 32) {
            if (LaneID() + i < partial)
                cub::ThreadStore<M>(devPointer + i, SMem[i]);
        }
        total -= partial;
        if (total <= 0)
            break;
		thread_mem_offset -= Items_per_warp;
		devPointer += partial;
	}
}

} //@warp

namespace block {

template<cub::CacheStoreModifier M, int BlockDim, int Items_per_block, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            const int size,
                                            int thread_mem_offset,
                                            const int total,
                                            T* __restrict__ SMem,
                                            T* __restrict__ devPointer) {
    int j = 0;
	while (true) {
		while (j < size && thread_mem_offset < Items_per_block) {
			SMem[thread_mem_offset] = Queue[j];
			j++;
            thread_mem_offset++;
		}
        __syncthreads();
		#pragma unroll
		for (int i = 0; i < Items_per_block; i += BlockDim) {
			const int index = threadIdx.x + i;
			if (index < total)
				cub::ThreadStore<M>(devPointer + index, SMem[index]);
		}
        total -= Items_per_block;
        if (total <= 0)
            break;
		thread_mem_offset -= Items_per_block;
		devPointer += Items_per_block;
        __syncthreads();
	}
}

} //@block
} //@dynamic
} //@data_movement
