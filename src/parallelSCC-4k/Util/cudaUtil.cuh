#pragma once

#include <iomanip>
//#include <limits>
#include <string>
#include <cuda_runtime.h>
#include "fUtil.h"
#include "printExt.h"

namespace scc4k{

#define cudaError(msg)		cudaDeviceSynchronize(); cudaUtil::_getLastCudaError (msg, __FILE__, __LINE__)


namespace cudaUtil {
	inline void _getLastCudaError(const char *errorMessage, const char *file, const int line);

	template<bool SORT = false, bool FAULT = true, typename T, typename R>
	void Compare(T* hostArray, R* devArray, const int size, std::string str = "", bool CHECK = true);

	template<class T>
	__global__ void scatterKernel(const int* toScatter, const int nof_items, T* Dest, const T value);

	template<class T>
	__global__ void scatterKernel(const int* toScatter, const int nof_items, T* Dest);

	template<class T>
	__global__ void fillKernel(T* toFill, const int nof_items, const T value);


	/*inline bool memInfoCUDA(size_t Req) {
		size_t free, total;
		cudaMemGetInfo(&free, &total);
		fUtil::memInfoPrint(total, free, Req);
		return free > Req;
	}*/

	inline float memInfoCUDA(size_t Req) {
		size_t free, total;
		cudaMemGetInfo(&free, &total);
		return (Req * 100) / free;
	}

	inline void cudaStatics() {
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		cudaError("ww");

		std::cout << std::endl
			<< "\t  Graphic Card: " << devProp.name << " (cc: " << devProp.major << "." << devProp.minor << ")" << std::endl
			<< "\t          # SM: " << devProp.multiProcessorCount
			<< "\t  Threads per SM: " << devProp.maxThreadsPerMultiProcessor
			<< "\t    Max Resident Thread: " << devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor << std::endl
			<< "\t   Global Mem.: " << devProp.totalGlobalMem / (1 << 20) << " MB"
			<< "\t     Shared Mem.: " << devProp.sharedMemPerBlock / 1024 << " KB"
			<< "\t               L2 Cache: " << devProp.l2CacheSize / 1024 << " KB" << std::endl
			<< "\tsmemPerThreads: " << devProp.sharedMemPerBlock / devProp.maxThreadsPerMultiProcessor << " Byte"
			<< "\t  regsPerThreads: " << devProp.regsPerBlock / devProp.maxThreadsPerMultiProcessor
			<< "\t              regsPerSM: " << devProp.regsPerBlock << std::endl << std::endl;

		#ifdef SCC_MAX_CONCURR_TH
		if (devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor != SCC_MAX_CONCURR_TH)
			error("Wrong SCC_MAX_CONCURR_TH configuration: " << devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor <<
					" vs. " << SCC_MAX_CONCURR_TH)
		#endif
	}



	inline int operator == (const int valueA, const int2 valueB) {
		return valueA == valueB.y;
	}

	inline int operator == (const int2 valueA, const int valueB) {
		return valueA.y == valueB;
	}

	inline int operator != (const int valueA, const int2 valueB) {
		return valueA != valueB.y;
	}

	inline int operator != (const int2 valueA, const int valueB) {
		return valueA.y != valueB;
	}

	inline bool operator< (const int2& valueA, const int2& valueB) {
		return valueA.x < valueB.x || (valueA.x == valueB.x && valueA.y < valueB.y );
	}

	inline std::ostream& operator<<(std::ostream& out, const int2& value) {
		out << value.y;
		return out;
	}

	inline void _getLastCudaError(const char *errorMessage, const char *file, const int line) {
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			std::cerr << std::endl << file << "(" << line << ") : getLastCudaError() CUDA error : " << errorMessage << " : (" << (int) err << ") " << cudaGetErrorString(err) << std::endl << std::endl;

			cudaError_t err2 = cudaSuccess;
			cudaDeviceReset();
			if(err2 != cudaSuccess)
			{
				std::cerr << "Error on resetting device!" << std::endl;
			}
			exit(EXIT_FAILURE);
		}
	}
	inline void reset()
	{
		cudaError_t err=cudaDeviceReset();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		std::cout << "Reset successfull" << std::endl;
		cudaError("reset");
	}
/*
	template<bool SORT, bool FAULT, typename T, typename R>
	void Compare(T* hostArray, R* devArray, const int size, std::string str, bool CHECK) {
		if (!CHECK)
			return;

		R* ArrayCMP = new R[size];
		cudaMemcpy(ArrayCMP, devArray, size * sizeof (R), cudaMemcpyDeviceToHost);
		cudaError("Copy To Host");

		fUtil::Compare<SORT, FAULT, T, R>(hostArray, ArrayCMP, size);
		//std::cout << str << " -> Correct" << std::endl;

		delete[] ArrayCMP;
	}

	template<typename T>
	void CompareSort(T* hostArray, T* devArray, const int size) {
		T* ArrayCMP = new T[size];
		cudaMemcpy(ArrayCMP, devArray, size * sizeof (T), cudaMemcpyDeviceToHost);
		cudaError("Copy To Host");

		std::sort(ArrayCMP, ArrayCMP + size);
		T* hostArrayTMP = new T[size];
		std::copy(hostArray, hostArray + size, hostArrayTMP);
		std::sort(hostArrayTMP, hostArrayTMP + size);

		for (int i = 0; i < size; ++i) {
			if (hostArrayTMP[i] != ArrayCMP[i])
				error("Sort Array Difference at: " << i << " -> ArrayA: " << hostArrayTMP[i] << " ArrayB: " << ArrayCMP[i]);
		}
		//std::cout << " -> Correct" << std::endl;

		delete[] ArrayCMP;
	}

	template<typename T, typename R>
	void CompareDistance(T* Array, R* devArray, const int size) {
		T* ArrayB = new T[size];
		cudaMemcpy(ArrayB, devArray, size * sizeof (R), cudaMemcpyDeviceToHost);
		cudaError("Copy To Host");

		for (int i = 0; i < size; ++i) {
			if (Array[i] != ArrayB[i] && !(Array[i] == std::numeric_limits<T>::max() && ArrayB[i] == std::numeric_limits<R>::max()) )
				error(" Distance Difference at: " << i << " -> ArrayA: " << Array[i] << " ArrayB: " << ArrayB[i]);
		}
	}
*/
#if defined(__NVCC__)
	template<class T>
	__global__ void scatterKernel(const int* toScatter, const int nof_items, T* Dest, const T value) {
		const int ID = blockIdx.x * blockDim.x + threadIdx.x;

		if (ID < nof_items)
			Dest[ toScatter[ID] ] = value;
	}
	template<class T>
	__global__ void scatterKernel(const int* toScatter, const int nof_items, T* Dest) {
		const int ID = blockIdx.x * blockDim.x + threadIdx.x;

		if (ID < nof_items)
			Dest[ toScatter[ID] ] = ID+1;
	}

	template<class T>
	__global__ void fillKernel(T* toFill, const int nof_items, const T value) {
		const int ID = blockIdx.x * blockDim.x + threadIdx.x;

		if (ID < nof_items)
			toFill[ ID ] = value;
	}
#endif
}

}
