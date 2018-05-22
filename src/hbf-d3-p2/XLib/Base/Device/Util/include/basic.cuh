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

/** @namespace basic
 *  provide basic cuda functions
 */
namespace basic {

/** @fn unsigned int WarpID()
 *  @brief return the warp ID within the block
 *
 *  Provide the warp ID within the current block.
 *  @return warp ID in the range 0 &le; ID &le; 32
 */
__device__ __forceinline__ unsigned int WarpID();

/** @fn int WarpBase()
 *  @brief return the warp ID within the block
 *
 *  Provide the warp ID within the current block.
 *  @return warp ID in the range 0 &le; ID &le; (BLOCKDIM / 32)
 */
template<int WARP_SZ>
__device__ __forceinline__ unsigned int WarpBase();

/** @fn T WarpBroadcast(T value, int predicate)
 *  @brief broadcast 'value' of the last lane that evaluates 'predicate' to true
 *
 *  @return 'value' of the last lane that evaluates 'predicate' to true
 */
template<typename T>
__device__ __forceinline__ T WarpBroadcast(T value, int predicate);

/** @fn void swap(T& A, T& B)
 *  @brief swap A and B
 */
template<typename T>
__device__ __forceinline__ void swap(T& A, T& B);

/** @fn log2(const int value)
 *  @brief calculate the integer logarithm of 'value'
 *  @return &lfloor; log2 ( value ) &rfloor;
 */
__device__ __forceinline__  int log2(const int value);

} //@basic

#include "impl/basic.i.cuh"
