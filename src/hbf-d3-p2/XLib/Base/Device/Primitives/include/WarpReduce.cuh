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

#include "../../../Host/BaseHost.hpp"

namespace primitives {

template<int WARP_SZ = 32>//!!!!!!!!!!!!! if WARP_SZ == 1
struct WarpReduce {
    static_assert(numeric::IS_POWER2<WARP_SZ>::value &&
                  WARP_SZ >= 1 && WARP_SZ <= 32,
                  PRINT_ERR(WarpReduce : WARP_SZ must be a power of 2 and
                            2 <= WARP_SZ <= 32));

    template<typename T>
    static __device__ __forceinline__ void Add(T& value);

    template<typename T>
    static __device__ __forceinline__ void Min(T& value);

    template<typename T>
    static __device__ __forceinline__ void Max(T& value);

    //--------------------------------------------------------------------------

    template<typename T>
    static __device__ __forceinline__ void AddBcast(T& value);

    template<typename T>
    static __device__ __forceinline__ void MinBcast(T& value);

    template<typename T>
    static __device__ __forceinline__ void MaxBcast(T& value);

    //--------------------------------------------------------------------------

    template<typename T>
    static __device__ __forceinline__ void Add(T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__ void Min(T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__ void Max(T& value, T* pointer);

    //--------------------------------------------------------------------------

    template<typename T>
    static __device__ __forceinline__ void AddAtom(T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__ void MinAtom(T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__ void MaxAtom(T& value, T* pointer);
};

} //@primitives

#include "impl/WarpReduce.i.cuh"
