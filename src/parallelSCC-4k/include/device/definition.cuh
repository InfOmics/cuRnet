#pragma once

#define l1(x) #x
#define l2(x) l1(x)

namespace scc4k{

#define Tid threadIdx.x

#if defined(__CUDACC__)
	#if ARCH >= 300 || __CUDA_ARCH__ >= 300
		#if ARCH == 300 || __CUDA_ARCH__ == 300
			//#pragma message("\n\nCompute Capability: 3\n")
			const int SMem_Per_SM =	49152;
		#elif ARCH == 350 || __CUDA_ARCH__ == 350
			//#pragma message("\n\nCompute Capability: 3.5\n")
			const int SMem_Per_SM =	49152;
		#elif ARCH == 370 || __CUDA_ARCH__ == 370
			//#pragma message("\n\nCompute Capability: 3.7\n")
			const int SMem_Per_SM =	114688;
		#elif ARCH == 500 || __CUDA_ARCH__ == 500
			//#pragma message("\n\nCompute Capability: 5.0\n")
			const int SMem_Per_SM =	65536;
		#elif ARCH == 520 || __CUDA_ARCH__ == 520
			//#pragma message("\n\nCompute Capability: 5.2\n")
			const int SMem_Per_SM = 98304;
		#elif ARCH == 530 || __CUDA_ARCH__ == 530
			//#pragma message("\n\nCompute Capability: 5.3\n")
			const int SMem_Per_SM = 65536;
		#elif ARCH == 600 || __CUDA_ARCH__ == 600
			//#pragma message("\n\nCompute Capability: 6.0\n")
			const int SMem_Per_SM = 65536;
		#elif ARCH == 610 || __CUDA_ARCH__ == 610
			//#pragma message("\n\nCompute Capability: 6.1\n")
			const int SMem_Per_SM = 98304;
		#elif ARCH == 620 || __CUDA_ARCH__ == 620
			//#pragma message("\n\nCompute Capability: 6.2\n")
			const int SMem_Per_SM = 65536;
		#elif ARCH == 700 || __CUDA_ARCH__ == 700
			//#pragma message("\n\nCompute Capability: 7.0\n")
			const int SMer_Per_SM = 98304;
		#else
			#pragma message("\n\nCompute Capability NOT included in list. Need to update " __FILE__ " at " l2(__LINE__) "\n\n")
		#endif
		const int      Thread_Per_SM  =  2048;
		const int    SMem_Per_Thread  =  SMem_Per_SM / Thread_Per_SM;
		const int IntSMem_Per_Thread  =  SMem_Per_Thread / 4;
		const int      SMem_Per_Warp  =  SMem_Per_Thread * 32;
		const int   IntSMem_Per_Warp  =  IntSMem_Per_Thread * 32;
		const int        MaxBlockDim  =  1024;
		const int         MemoryBank  =  32;
	#else
		#error message("\n\nCompute Capability NOT supported (<3)\n")
	#endif
#endif

#define                  MIN_V(a, b)	((a) > (b) ? (b) : (a))
#define     MAX_CONCURR_BL(BlockDim)	( SCC_MAX_CONCURR_TH / (BlockDim) )
#define     SMem_Per_Block(BlockDim)	( MIN_V( SMem_Per_Thread * (BlockDim) , 49152) )
#define  IntSMem_Per_Block(BlockDim)	( MIN_V( IntSMem_Per_Thread * (BlockDim) , 49152 / 4) )

}
