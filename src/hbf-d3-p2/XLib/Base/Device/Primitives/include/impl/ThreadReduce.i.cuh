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
#include "../cuda_functional.cuh"

namespace primitives {
using namespace cuda_functional;

namespace {

using namespace numeric;

template<int SIZE, typename T, binary_op<T> BinaryOP, int STRIDE = 1>
struct ThreadReduceSupport {
    static_assert(numeric::IS_POWER2<SIZE>::value,
                  PRINT_ERR(ThreadReduce : SIZE must be a power of 2));

    __device__ __forceinline__ static void UpSweepLeft(T (&Array)[SIZE]) {
        #pragma unroll
        for (int INDEX = 0; INDEX < SIZE; INDEX += STRIDE * 2)
            Array[INDEX] = BinaryOP(Array[INDEX], Array[INDEX + STRIDE]);
        ThreadReduceSupport<SIZE, T, BinaryOP, STRIDE * 2>::UpSweepLeft(Array);
    }

    __device__ __forceinline__ static void UpSweepRight(T (&Array)[SIZE]) {
        #pragma unroll
        for (int INDEX = STRIDE - 1; INDEX < SIZE; INDEX += STRIDE * 2)
            Array[INDEX + STRIDE] = BinaryOP(Array[INDEX], Array[INDEX + STRIDE]);
        ThreadReduceSupport<SIZE, T, BinaryOP, STRIDE * 2>::UpSweepRight(Array);
    }
};

template<int SIZE, typename T, binary_op<T> BinaryOP>
struct ThreadReduceSupport<SIZE, T, BinaryOP, SIZE> {
    __device__ __forceinline__ static void UpSweepLeft(T (&Array)[SIZE]) {}
    __device__ __forceinline__ static void UpSweepRight(T (&Array)[SIZE]) {}
};

} //@anonymous

//==============================================================================

namespace ThreadReduce {

template<typename T, int SIZE>
__device__ __forceinline__ static void Add(T (&Array)[SIZE]) {
	ThreadReduceSupport<SIZE, T, plus<T>>::UpSweepLeft(Array);
}

template<typename T, int SIZE>
__device__ __forceinline__ static void Min(T (&Array)[SIZE]) {
	ThreadReduceSupport<SIZE, T, _min<T>>::UpSweepLeft(Array);
}

template<typename T, int SIZE>
__device__ __forceinline__ static void Max(T (&Array)[SIZE]) {
	ThreadReduceSupport<SIZE, T, _max<T>>::UpSweepLeft(Array);
}

template<typename T, int SIZE>
__device__ __forceinline__ static void LogicAnd(T (&Array)[SIZE]) {
	ThreadReduceSupport<SIZE, T, logical_and<T>>::UpSweepLeft(Array);
}

} //@ThreadReduce
} //@primitives
