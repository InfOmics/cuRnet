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

#define REQUIRE_ARCH_DEF
#include "../../Util/include/definition.cuh"

namespace data_movement {
namespace warp {

/**
* Always ordered
*/
template<int SIZE,
         cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         typename T>
void __device__ __forceinline__ SharedToGlobal(T* __restrict__ SMem,
                                               T* __restrict__ Pointer);

/**
* Always ordered
*/
template<int SIZE,
         cub::CacheLoadModifier M = cub::CacheLoadModifier::LOAD_DEFAULT,
         typename T>
void __device__ __forceinline__ GlobalToShared(T* __restrict__ Pointer,
                                              T* __restrict__ SMem);

//------------------------------------------------------------------------------

namespace ordered {

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
        typename T, int Items_per_warp = SMem_Per_Warp<T>::value, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ SMem,
                                            T* __restrict__ Pointer);

template<cub::CacheLoadModifier M = cub::CacheLoadModifier::LOAD_DEFAULT,
        typename T, int Items_per_warp = SMem_Per_Warp<T>::value, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T* __restrict__ SMem,
                                            T (&Queue)[SIZE]);

/**
* Can involve bank conflict
*/
template<typename T, int SIZE>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem);

/**
* Can involve bank conflict
*/
template<typename T, int SIZE>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]);

} //@ordered

//------------------------------------------------------------------------------

namespace unordered {

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ Pointer);

template<cub::CacheLoadModifier M = cub::CacheLoadModifier::LOAD_DEFAULT,
         typename T, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T (&Queue)[SIZE]);

template<int SIZE, typename T>
void __device__ __forceinline__ RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem);

template<int SIZE, typename T>
void __device__ __forceinline__ SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]);

} //@unordered
} //@warp
} //@data_movement

#include "impl/DataMovementWarp.i.cuh"
#include "impl/DataMovementWarpUnordered.i.cuh"
#include "impl/DataMovementWarpOrdered.i.cuh"
