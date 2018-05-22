
namespace scc4k{

//if BLOCK_KERNEL -> SAFE = 0
//if SYNC -> DYNAMIC PARALLELLISM = 0

#define HASHTABLE_BLOCK_POS  0
#define END_OF_HASHTABLE	(4096 * 8)	// 8: long long int size
#define       F2Size_POS	END_OF_HASHTABLE
#define         TEMP_POS	(F2Size_POS + 4)
#define     END_TEMP_POS	(TEMP_POS + 32 * 4)

#define    FRONTIER_SIZE	(((49152 - END_TEMP_POS) / 2) - 2)		//-2 align
#define     F1_BLOCK_POS	(END_TEMP_POS)
#define     F2_BLOCK_POS	(F1_BLOCK_POS + FRONTIER_SIZE)


template<int BlockDim, int WARP_SZ, int DUP_REM>
__device__ __forceinline__ void BFS_BlockKernelB (	      int* __restrict__		devNode,
													      int* __restrict__		devEdge,
													   dist_t* __restrict__ 	devDistance,
														  int* __restrict__ 	F1,
														  int* __restrict__ 	F2,
														  int* __restrict__ 	devF2,
														  int* __restrict__		F2SizePtr,
													      int FrontierSize, int level,
													volatile long long int*		HashTable) {

		int Queue[REG_QUEUE];
		int founds = 0;
		for (int t = Tid >> _Log2<WARP_SZ>::VALUE; t < FrontierSize; t += BlockDim / WARP_SZ) {
			const int index = F1[t];
			const int start = devNode[index];
			int end = devNode[index + 1];

			EdgeVisit<BlockDim, WARP_SZ, DUP_REM * 2>(devEdge, devDistance, NULL, NULL, index, start, end, Queue, founds, level, HashTable);
		}
		/*const int indexT1 = Tid >> _Log2<WARP_SZ>::VALUE;
		#pragma unroll
		for (int t = 0; t < DIV(BLOCK_FRONTIER_LIMIT, BlockDim); t++) {
			const int indexT = indexT1 + t * (BlockDim / WARP_SZ);
			if (indexT < FrontierSize) {
				const int index = ThreadLoadV<int, LOAD_MODE>(F1, indexT);
				const int start = ThreadLoadV<int, LOAD_MODE>(devNode, index);
				int end = ThreadLoadV<int, LOAD_MODE>(devNode, index + 1);

				EdgeVisit<BlockDim, WARP_SZ, DUP_REM>(devEdge, devDistance, devF2, F2SizePtr, index, start, end, Queue, founds, level, HashTable);
			}
		}*/

		int WarpPos, n, total;
		singleblockQueueAdd(founds, F2SizePtr, WarpPos, n, total, level, (int*) &SMem[TEMP_POS]);

		if (WarpPos + total >= BLOCK_FRONTIER_LIMIT) {
			if (WarpPos < BLOCK_FRONTIER_LIMIT)
				SMem[0] = WarpPos;
			writeOPT<SIMPLE, STORE_DEFAULT>(devF2, Queue, founds, WarpPos, n, total);
		} else {
			writeOPT<SIMPLE, STORE_DEFAULT>(F2, Queue, founds, WarpPos, n, total);
		}
}



#define fun(a)		BFS_BlockKernelB<1024, (a), DUP_REM>\
							(devNodes, devEdges, devDistance, SMemF1, SMemF2, devF2, F2SizePtr, FrontierSize, level, HashTable);

template<int DUP_REM>
__global__ void BFS_BlockKernel (		  int* __restrict__		devNodes,
										  int* __restrict__		devEdges,
									   dist_t* __restrict__ 	devDistance,
										  int* __restrict__ 	devF1,
										  int* __restrict__	 	devF2,
									               const int devF1Size) {

	volatile long long int* HashTable;
	if (DUP_REM)
		HashTable = (volatile long long int*) SMem;

	int level = devLevel;
	int FrontierSize = devF1Size;
	
//	if (Tid == 0)
	//	printf("END_OF_HASH %d     END_TEMP_POS %d   FRONTIER_SIZE %d  F1 %d       F2 %d\n", END_OF_HASHTABLE, END_TEMP_POS, FRONTIER_SIZE,  F1_BLOCK_POS, F2_BLOCK_POS);
	
	int* SMemF1 = devF1;
	int* SMemF2 = (int*) &SMem[F2_BLOCK_POS];
	int* F2SizePtr = (int*) &SMem[F2Size_POS];
		
	int size = logValueDevice<1024, MIN_VW, MAX_VW>(FrontierSize);
	
	def_SWITCH(size);
	
	SMemF1 = SMemF2;
	SMemF2 = (int*) &SMem[F1_BLOCK_POS];
	level++;
	__syncthreads();
	FrontierSize = F2SizePtr[0];
				
	while (FrontierSize && FrontierSize < BLOCK_FRONTIER_LIMIT) {
		int size = logValueDevice<1024, MIN_VW, MAX_VW>(FrontierSize);
		
		def_SWITCH(size);
		
		swapDev(SMemF1, SMemF2);
		level++;
		__syncthreads();
		FrontierSize = F2SizePtr[0];		
	}
	// ----------------------- ENDING PHASE-------------------------------
	__syncthreads();
	if (Tid == 0)
		devF2Size[level & 3] = FrontierSize;
	if ( FrontierSize == 0 )
		return;

	if (Tid == 32)
		devLevel = level;
	const int totan_on_SMem = SMem[0];
	#pragma unroll
	for (int i = 0; i < DIV(BLOCK_FRONTIER_LIMIT, 1024); i++) {
		const int index = Tid + i * 1024;
		if (index < totan_on_SMem)
			devF2[index] = SMem[index];
	}
	
	//if (Tid == 0)
	//	printf("SIZE: %d\n", FrontierSize);
}

#undef fun

}
