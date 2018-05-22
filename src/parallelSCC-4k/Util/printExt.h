#pragma once

#include <iostream>
#include <string>
#include <limits>
#include <cuda_runtime.h>
#include "cudaUtil.cuh"



namespace std {
	template<> class numeric_limits<int2> {
		public:
        static int2 max() {return make_int2(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());}
	};
	
	template<> class numeric_limits<int3> {
		public:
        static int3 max() {return make_int3(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max());}
	};
	
	#if __cplusplus < 199711L && ! __GXX_EXPERIMENTAL_CXX0X__
	template<typename T>
	std::string to_string(T x) {
		std::stringstream ss;
		ss << x;
		return ss.str();
	}
	#endif
}


namespace scc4k{

namespace printExt {

template<class T>
void printArray(T Array[], const int size, std::string text = "", bool debug = true, const char sep = ' ', T inf = std::numeric_limits<T>::max());

template<>		
void printArray<int2>(int2 Array[], const int size, std::string text, bool debug, const char sep, int2 inf);

template<>		
void printArray<int3>(int3 Array[], const int size, std::string text, bool debug, const char sep, int3 inf);

template<class T>
void printContMatrix(T** Matrix, const int ROW, const int COL, std::string text, bool debug = true, T inf = std::numeric_limits<T>::max());
	
	

template<class T>
inline void printArray(T Array[], const int size, std::string text, bool debug, const char sep, T inf) {
	if (!debug)
		return;

	std::cout << text;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i] == inf) ? "inf" : std::to_string(Array[i])) << sep;
	std::cout << std::endl << std::endl;
}
	
template<>		
inline void printArray<int2>(int2 Array[], const int size, std::string text, bool debug, const char sep, int2 inf) {
	if (!debug)
		return;

	std::cout << text;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].x == inf.x) ? "inf" : std::to_string(Array[i].x)) << '\t';
	std::cout << std::endl;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].x == inf.x) ? "inf" : std::to_string(Array[i].y)) << '\t';
	std::cout << std::endl << std::endl;
}

template<>		
inline void printArray<int3>(int3 Array[], const int size, std::string text, bool debug, const char sep, int3 inf) {
	if (!debug)
		return;

	std::cout << text;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].x == inf.x) ? "inf" : std::to_string(Array[i].x)) << '\t';
	std::cout << std::endl;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].x == inf.x) ? "inf" : std::to_string(Array[i].y)) << '\t';
	std::cout << std::endl;
	for (int i = 0; i < size; i++)
		std::cout << ((Array[i].x == inf.x) ? "inf" : std::to_string(Array[i].z)) << '\t';
	std::cout << std::endl << std::endl;
}

template<class T>
void printContMatrix(T** Matrix, const int ROW, const int COL, std::string text, bool debug, T inf) {
	if (!debug)
		return;

	std::cout << text;
	for (int i = 0; i < ROW; i++)
		printArray(Matrix[i * COL], COL, "\n", true, inf, '\t');
	std::cout << std::endl << std::endl;
}

#if defined(__NVCC__)

	template<class T>
	void printCudaArray(T* devArray, const int size, std::string text = "", bool debug = true, const char sep = ' ', T inf = std::numeric_limits<T>::max());
	
	template<class T>
	void printCudaArray(T* devArray, const int size, std::string text, bool debug, const char sep, T inf) {
		if (!debug)
			return;

		T* hostArray = new T[size];
		cudaMemcpy(hostArray, devArray, size * sizeof (T), cudaMemcpyDeviceToHost);
		//cudaError("Copy To Host");

		printArray<T>(hostArray, size, text, debug, sep, inf);
		delete hostArray;
	}
#endif
}

}
