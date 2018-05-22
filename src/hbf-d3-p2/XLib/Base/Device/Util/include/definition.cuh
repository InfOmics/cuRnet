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

#define l1(x) #x
#define l2(x) l1(x)

#include "../../../Host/include/Numeric.hpp"
using namespace numeric;

const int        THREAD_PER_SM  =  2048;
const int         MAX_BLOCKDIM  =  1024;
const int MAX_SHARED_PER_BLOCK  =  49152;
const int         CONSTANT_MEM  =  49152;
const int          MEMORY_BANK  =  32;
const int            WARP_SIZE  =  32;

template<typename T>
struct Max_SMem_Per_Block {
	static const int value = MAX_SHARED_PER_BLOCK / sizeof(T);
};

#if defined(__CUDACC__)
	#if __CUDA_ARCH__  >= 300 || ARCH >= 300
		#if __CUDA_ARCH__ == 300 || ARCH == 300
			//#pragma message("\n\nCompute Capability: 3\n")
			const int SMEM_PER_SM =	49152;
			const int RESIDENT_BLOCKS_PER_SM = 16;

		#elif __CUDA_ARCH__ == 320 || ARCH == 320
			//#pragma message("\n\nCompute Capability: 3.2\n")
			const int SMEM_PER_SM =	49152;
			const int RESIDENT_BLOCKS_PER_SM = 16;

		#elif __CUDA_ARCH__ == 350 || ARCH == 350
			//#pragma message("\n\nCompute Capability: 3.5\n")
			const int SMEM_PER_SM =	49152;
			const int RESIDENT_BLOCKS_PER_SM = 16;

		#elif __CUDA_ARCH__ == 370 || ARCH == 370
		//	#pragma message("\n\nCompute Capability: 3.7\n")
			const int SMEM_PER_SM =	114688;
			const int RESIDENT_BLOCKS_PER_SM = 16;

		#elif __CUDA_ARCH__ == 500 || ARCH == 500
		//	#pragma message("\n\nCompute Capability: 5.0\n")
			const int SMEM_PER_SM =	65536;
			const int RESIDENT_BLOCKS_PER_SM = 32;

		#elif __CUDA_ARCH__ == 520 || ARCH == 520
		//	#pragma message("\n\nCompute Capability: 5.2\n")
			const int SMEM_PER_SM = 98304;
			const int RESIDENT_BLOCKS_PER_SM = 32;

		#elif __CUDA_ARCH__ == 530 || ARCH == 530
			//#pragma message("\n\nCompute Capability: 5.3\n")
			const int SMEM_PER_SM = 65536;
			const int RESIDENT_BLOCKS_PER_SM = 32;
			
		#elif __CUDA_ARCH__ == 600 || ARCH == 600
			//#pragma message("\n\nCompute Capability: 6.0\n")
			const int SMEM_PER_SM = 65536;
			const int RESIDENT_BLOCKS_PER_SM = 32;
			
		#elif __CUDA_ARCH__ == 610 || ARCH == 610
			//#pragma message("\n\nCompute Capability: 6.1\n")
			const int SMEM_PER_SM = 98304;
			const int RESIDENT_BLOCKS_PER_SM = 32;
			
		#elif __CUDA_ARCH__ == 620 || ARCH == 620
			//#pragma message("\n\nCompute Capability: 6.2\n")
			const int SMEM_PER_SM = 65536;
			const int RESIDENT_BLOCKS_PER_SM = 32;
			
		#elif __CUDA_ARCH__ == 700 || ARCH == 700
			//#pragma message("\n\nCompute Capability: 7.0\n")
			const int SMEM_PER_SM = 98304;
			const int RESIDENT_BLOCKS_PER_SM = 32;
		#else
			#pragma message("\n\nCompute Capability NOT included in list. Need to update " __FILE__ " at " l2(__LINE__) "\n\n")
		#endif

		template<typename T>
		struct SMem_Per_Thread {
			static const int value = (SMEM_PER_SM / THREAD_PER_SM) / sizeof(T);
		};
		template<typename T>
		struct SMem_Per_Warp {
			static const int value = SMem_Per_Thread<T>::value * WARP_SIZE;
		};
		template<typename T, int BlockDim>
		struct SMem_Per_Block {
			static const int value = MIN< SMem_Per_Thread<T>::value * BlockDim, (int) (MAX_SHARED_PER_BLOCK / sizeof(T)) >::value;
		};
	#else
		#pragma error(ERR_START "Unsupported Compute Cabalitity" ERR_END)
	#endif
#endif

#if defined(SM)
	const int           NUM_SM  =  SM;
	const int RESIDENT_THREADS  =  SM * THREAD_PER_SM;
	const int   RESIDENT_WARPS  =  RESIDENT_THREADS / 32;

    template<int BlockDim>
    struct RESIDENT_BLOCKS {
        static const int value = RESIDENT_THREADS / BlockDim;
    };
#endif
