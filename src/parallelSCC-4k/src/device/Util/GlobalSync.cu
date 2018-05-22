#pragma once

#include <cub/cub.cuh>

namespace scc4k{

__device__ unsigned int GSync[MAX_CONCURR_BL(BLOCKDIM)];

__global__ void GReset() {
	if (Tid < MAX_CONCURR_BL(BLOCKDIM))
		GSync[Tid] = 0;
}

__device__  __forceinline__ void GlobalSync() {
	volatile unsigned *VolatilePtr = GSync;
	__syncthreads();

	if (blockIdx.x == 0) {
		if (Tid == 0)
			VolatilePtr[blockIdx.x] = 1;
		//__syncthreads();
	
		if (Tid < MAX_CONCURR_BL(BLOCKDIM))
			while ( cub::ThreadLoad<cub::LOAD_CG>(GSync + Tid) == 0 );

		__syncthreads();

		if (Tid < MAX_CONCURR_BL(BLOCKDIM))
			VolatilePtr[Tid] = 0;
	}
	else {
		if (Tid == 0) {
			VolatilePtr[blockIdx.x] = 1;
			while (cub::ThreadLoad<cub::LOAD_CG>(GSync + blockIdx.x) == 1);
		}
		__syncthreads();
	}
}

}
