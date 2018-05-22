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

#include <iostream>
#include <string>
#include <limits>
#if __NVCC__
    #include <cuda_runtime.h>
    #include "../../Device/BaseDevice.cuh"
#endif

namespace std {
	#if __NVCC__
	template<> class numeric_limits<int2> {
		public:
        static int2 max() {return make_int2(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());}
	};

	template<> class numeric_limits<int3> {
		public:
        static int3 max() {return make_int3(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max());}
	};
	#endif

	#if __cplusplus < 199711L && ! __GXX_EXPERIMENTAL_CXX0X__
	template<typename T>
	std::string to_string(T x) {
		std::stringstream ss;
		ss << x;
		return ss.str();
	}
	#endif
}

namespace printExt {
namespace host {

template<class T>
void printArray(T Array[], const int size, std::string text = "", const char sep = ' ', T inf = std::numeric_limits<T>::max());

template<class T>
void printMatrix(T** Matrix, const int ROW, const int COL, std::string text = "", T inf = std::numeric_limits<T>::max());

} //@Host


#if  defined(__NVCC__)

namespace device {

template<class T>
void printArray(T* devArray, const int size, std::string text = "", const char sep = ' ', T inf = std::numeric_limits<T>::max());

template<>
void printArray<int2>(int2 Array[], const int size, std::string text, const char sep, int2 inf);

template<>
void printArray<int3>(int3 Array[], const int size, std::string text, const char sep, int3 inf);

} //@device
#endif
} //@printExt

#include "impl/printExt.i.hpp"
