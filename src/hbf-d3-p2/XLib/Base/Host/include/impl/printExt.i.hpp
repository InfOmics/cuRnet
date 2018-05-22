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
#if defined(__NVCC__)
	#include <vector_types.h>	//int2
#endif
#include <type_traits>

namespace printExt {
namespace host {

#if defined(__MINGW32__) || defined(__MINGW64__)
    namespace std {
        template <typename T>
        std::string to_string(const T& n ){
            std::ostringstream stm;
            stm << n ;
            return stm.str() ;
        }
    }
#endif

template<class T>
inline void printArray(T Array[], const int size, std::string text, const char sep, T inf) {
	std::cout << text;

    for (int i = 0; i < size; i++)
        std::cout << ((Array[i] == inf) ? "inf" : std::to_string(Array[i])) << sep;
	std::cout << std::endl << std::endl;
}

template<class T>
void printMatrix(T** Matrix, const int ROW, const int COL, std::string text, T inf) {
	std::cout << text;
	for (int i = 0; i < ROW; i++)
		printArray(Matrix[i * COL], COL, "\n", true, inf, '\t');
	std::cout << std::endl << std::endl;
}

} //@Host

#if defined(__NVCC__)
namespace device {

template<>
inline void printArray<int2>(int2 Array[], const int size, std::string text, const char sep, int2 inf) {
	std::cout << text;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].x == inf.x) ? "inf" : std::to_string(Array[i].x)) << sep;
	std::cout << std::endl;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].y == inf.y) ? "inf" : std::to_string(Array[i].y)) << sep;
	std::cout << std::endl << std::endl;
}

template<>
inline void printArray<int3>(int3 Array[], const int size, std::string text,__attribute__((unused)) const char sep, int3 inf) {
	std::cout << text;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].x == inf.x) ? "inf" : std::to_string(Array[i].x)) << '\t';
	std::cout << std::endl;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].y == inf.y) ? "inf" : std::to_string(Array[i].y)) << '\t';
	std::cout << std::endl;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].z == inf.z) ? "inf" : std::to_string(Array[i].z)) << '\t';
	std::cout << std::endl << std::endl;
}

template<class T>
void printCudaArray(T* devArray, const int size, std::string text, const char sep, T inf) {
    T* hostArray = new T[size];
    cudaMemcpy(hostArray, devArray, size * sizeof (T), cudaMemcpyDeviceToHost);
    __CUDA_ERROR("Copy To Host");

    printArray<T>(hostArray, size, text, sep, inf);
    delete hostArray;
}
} //@device
#endif
} //@printExt
