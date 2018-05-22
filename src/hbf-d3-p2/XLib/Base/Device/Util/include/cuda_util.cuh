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

#include <string>
#include <cuda_runtime.h>

#define __CUDA_ERROR(msg)                                                       \
                    {                                                           \
                        cudaDeviceSynchronize();                                \
                        cuda_util::__getLastCudaError (msg, __FILE__, __LINE__);\
                    }

inline std::ostream& operator << (std::ostream& out, const int2& value);
inline bool operator== (const int2& A, const int2& B);
inline bool operator!= (const int2& A, const int2& B);
inline bool operator<  (const int2& A, const int2& B);
inline bool operator<= (const int2& A, const int2& B);
inline bool operator>  (const int2& A, const int2& B);
inline bool operator>= (const int2& A, const int2& B);

namespace cuda_util {

inline void __getLastCudaError(const char *errorMessage,
                               const char *file,
                               const int line);

class deviceProperty {
	public:
		static int getNum_of_SMs();
	private:
		static int NUM_OF_STREAMING_MULTIPROCESSOR;
};

template<bool FAULT = true, typename T, typename R>
void Compare(T* hostArray, R* devArray, const int size);

template<bool FAULT = true, typename T, typename R>
bool Compare(T* hostArray, R* devArray, const int size, bool (*cmpFunction)(T, R));

template<bool FAULT = true, typename T, typename R>
void CompareSort(T* hostArray, R* devArray, const int size);

namespace NVTX {

const int GREEN = 0x0000ff00, BLUE = 0x000000ff, YELLOW = 0x00ffff00,
          PURPLE = 0x00ff00ff, CYAN = 0x0000ffff, RED = 0x00ff0000,
          WHITE = 0x00ffffff;

void PushRange(std::string s, const int color);
void PopRange();

} //@NNVTX


bool memInfoCUDA(size_t Req);
void memCheckCUDA(size_t Req);
void cudaStatics();

template<typename T>
__global__ void scatter(const int* __restrict__ toScatter,
                        const int scatter_size,
                        T*__restrict__ Dest,
                        const T value);

template<typename T>
__global__ void fill(T* devArray, const int fill_size, const T value);

template<typename T>
__global__ void fill2(T* devArray, const int fill_size, const T value);

template <typename T>
__global__ void fill(T* devMatrix, const int n_of_rows, const int n_of_columns,
                     const T value, int integer_pitch = 0);

} //@cuda_util

#include "impl/cuda_util.i.cuh"
