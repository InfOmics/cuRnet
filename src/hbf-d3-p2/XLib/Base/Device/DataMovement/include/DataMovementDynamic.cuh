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

#include <cub/cub.cuh>

namespace data_movement {
namespace dynamic {
namespace thread {

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            const int size,
                                            T* __restrict__ devPointer);

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal_Unroll(T (&Queue)[SIZE],
                                            const int size,
                                            T* __restrict__ devPointer);

} // @thread

namespace warp {

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         int Items_per_warp, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            const int size,
                                            int thread_mem_offset,
                                            const int total,
                                            T* __restrict__ SMem,
                                            T* __restrict__ devPointer);

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         int Items_per_warp, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal_Min(T (&Queue)[SIZE],
                                                const int size,
                                                int thread_mem_offset,
                                                const int total,
                                                T* __restrict__ SMem,
                                                T* __restrict__ devPointer);


template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         int Items_per_warp, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal_Unroll(T (&Queue)[SIZE],
                                            const int size,
                                            int thread_mem_offset,
                                            const int total,
                                            T* __restrict__ SMem,
                                            T* __restrict__ devPointer);
} //@warp

namespace block {

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         int BlockDim, int Items_per_warp, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            const int size,
                                            int thread_mem_offset,
                                            const int total,
                                            T* __restrict__ SMem,
                                            T* __restrict__ devPointer) ;

} //@block
} //@dynamic
} //@data_movement

#include "impl/DataMovementDynamic.i.cuh"
