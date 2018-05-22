#pragma once
#include "assert.h"

namespace scc4k{

template<int BlockDim, bool forward, VisitType visitType>
__global__ void DynamicKernel(const int V, 	int* __restrict__ devEdge,
								dist_t* devDistance, int* devF2,
								const int start, const int end,
								const int level,
								color_t*  __restrict__ color,
								const int colorToConfront, const int colorToTakeValue
							) {

	int* devF2SizePrt = &devF2Size[level & 3];
	int k = start+blockIdx.x * BlockDim + Tid;

	int Queue[REG_QUEUE];
	int founds = 0;

	bool flag = true;
	while (flag) {
		while (k < end && founds < REG_QUEUE) {
			const int dest = devEdge[k];

			KVisit<BlockDim, false, forward, visitType>(V, dest, devDistance, Queue, founds, level, NULL,
				color, colorToConfront, colorToTakeValue);
			k += gridDim.x * BlockDim;
		}
		if (__any(founds >= REG_QUEUE)) {
			FrontierWrite::Write<BlockDim, FrontierWrite::SHARED_WARP>(devF2, devF2SizePrt, Queue, founds);
			founds = 0;
		} else
			flag = false;
	}
	FrontierWrite::Write<BlockDim, FrontierWrite::SHARED_WARP>(devF2, devF2SizePrt, Queue, founds);
}


template<int BlockDimDyn, int WARP_SZ, int SINGLE, bool forward, VisitType visitType, bool DYNAMIC_PARALLELISM>
__device__ __forceinline__ void DynamicParallelism(const int V, int*      __restrict__   devEdge,
													dist_t*   __restrict__   devDistance,
													color_t*  __restrict__   color,
													int*      __restrict__   devF2,
													int start, int& end,
													const int level,
													const int colorToConfront, const int colorToTakeValue) {
	if (DYNAMIC_PARALLELISM) {
		const int degree = end - start;
		if (degree >= THRESHOLD_G) {
			if (SINGLE || (Tid & _Mod2<WARP_SZ>::VALUE) == 0) {
				//const int DynGridDim = (degree + (BlockDimDyn * EDGE_PER_TH) - 1) >> (_Log2<BlockDimDyn>::VALUE + _Log2<EDGE_PER_TH>::VALUE);
				DynamicKernel<BlockDimDyn, forward, visitType><<<LAUNCHED_BLOCKS, BlockDimDyn, SMem_Per_Block(BlockDimDyn) >>> (V, devEdge, devDistance, devF2, start, end, level, color, colorToConfront, colorToTakeValue);
			}
			end = INT_MIN;
		}
	}
}

}
