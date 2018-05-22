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
struct WarpInclusiveScan {
    /// @cond
    static_assert(numeric::IS_POWER2<WARP_SZ>::value &&
                  WARP_SZ >= 1 && WARP_SZ <= 32,
                  PRINT_ERR(WarpInclusiveScan : WARP_SZ must be a power of 2 and
                            2 <= WARP_SZ <= 32));
    /// @endcond

    template<typename T>
    static __device__ __forceinline__ void Add(T& value);

    template<typename T>
    static __device__ __forceinline__ void Add(T& value, T& total);

    template<typename T>
    static __device__ __forceinline__ void AddBcast(T& value, T& pointer);

    template<typename T>
    static __device__ __forceinline__ void Add(T& value, T* pointer);
};

//------------------------------------------------------------------------------

/** \struct WarpExclusiveScan WarpScan.cuh
 *  \brief Support structure for warp-level exclusive scan
 *  <pre>
 *  Input:  1 2 3 4
 *  Output: 0 1 3 6 (10)
 *  </pre>
 *  \callergraph \callgraph
 *  @pre WARP_SZ must be a power of 2 in the range 1 &le; WARP_SZ &le; 32
 *  @tparam WARP_SZ     split the warp in WARP_SIZE / WARP_SZ groups and
 *                      perform the exclusive prefix-scan in each groups.
 *                      Require log2 ( WARP_SZ ) steps
 */
template<int WARP_SZ = 32>
struct WarpExclusiveScan {
    /// @cond
    static_assert(numeric::IS_POWER2<WARP_SZ>::value &&
                  WARP_SZ >= 2 && WARP_SZ <= 32,
                  PRINT_ERR(WarpExclusiveScan : WARP_SZ must be a power of 2 and
                            2 <= WARP_SZ <= 32));
    /// @endcond

    template<typename T>
    static __device__ __forceinline__ void Add(T& value);

    /** \fn void Add(T& value, T& total)
     *  \brief warp sum
     *  @param[in] value    input parameter for each thread
     *  @param[out] total   total sum of all values
     *  \warning only the last thread in the WARP_SZ group has the total sum
     */
    template<typename T>
    static __device__ __forceinline__ void Add(T& value, T& total);

    template<typename T>
    static __device__ __forceinline__ void AddBcast(T& value, T& total);

    template<typename T>
    static __device__ __forceinline__ void Add(T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__ T AddAtom(T& value, T* pointer);

    //template<typename T>
    //static __device__ __forceinline__ void Add(T& inPointer, T* totalPointer);
};

} //@primitives

#include "impl/WarpScan.i.cuh"
