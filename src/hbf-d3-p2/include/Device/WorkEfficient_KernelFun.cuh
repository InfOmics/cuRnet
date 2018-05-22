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

#include "Device/HBFGraph.cuh"
//#include "config.cuh"
#include <cub/cub.cuh>
#include "XLib.hpp"


namespace kernels {

__device__ __forceinline__ void KRelax(	const EdgeT dest,
                                        const node_t index,
                                        const weight_t node_weight,
					                    hdist_t* devDistance,
                                        node_t* Queue,
                                        int& founds,
                                        const int level,
					                    long long unsigned* d_distances) {
    if (dest.y == 0.0f) {
        DistPath   tmp = { index, node_weight };
        auto dist_path = reinterpret_cast<long long unsigned&>(tmp);

        long long unsigned assumed, old = d_distances[dest.x];
        auto old_dist_path = reinterpret_cast<DistPath&>(old);
        //printf("src %d, src_weight %f \t d: %d \t w: %f\n",
        //        index, node_weight, dest.x, old_dist_path.dist);
        if (old_dist_path.dist <= node_weight)
            return; //weight(dest) <= tentative
        do {
            assumed       = old;
            old           = atomicCAS(d_distances + dest.x, assumed, dist_path);
            old_dist_path = reinterpret_cast<DistPath&>(old);
            //printf("    src %d, src_weight %f \t d: %d \t w: %f\n",
            //        index, node_weight, dest.x, old_dist_path.dist);
            if (old_dist_path.dist <= node_weight)
                return; //weight(dest) <= tentative 
        } while (old != assumed);
        Queue[founds++] = dest.x;
    }
    else {
        float  tent_weight = node_weight + dest.y;
        DistPath       tmp = { index, tent_weight };
        auto     dist_path = reinterpret_cast<long long unsigned&>(tmp);
        auto       old_min = atomicMin(d_distances + dest.x, dist_path);
        auto old_dist_path = reinterpret_cast<DistPath&>(old_min);
        if (old_dist_path.dist > tent_weight)
            Queue[founds++] = dest.x;
    }

    //printf("%f %d    \t %lld\n", tmp.dist, tmp.parent , dist_path);
    //printf("e (%d, %d) \t w: %f \t o: %f \t t: %f \t %d \t p: %d\n",
    //        index, dest.x, dest.y, old_dist_path.dist, tent_weight,
    //        old_dist_path.dist > tent_weight, old_dist_path.parent);
}

using namespace data_movement::dynamic;

template<int VW_SIZE, int SIZE>
__device__ __forceinline__ void EdgeVisit(	EdgeT* __restrict__ devEdges,
                                            hdist_t* __restrict__ devDistances,
											node_t* __restrict__ devF2,
                                            int* __restrict__ devF2SizePrt,
											const node_t index,
                                            const weight_t nodeWeight,
                                            const edge_t start,
                                            edge_t end,
											node_t (&Queue)[SIZE],
                                            int& founds,
                                            const int level,
						                    long long unsigned* __restrict__ d_distances) {
/*if (!SAFE)
	for (int k = start + (threadIdx.x & MOD2<VW_SIZE>::value); k < end; k += VW_SIZE) {
		const auto dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdges + k);		// X: edge index	Y: edge weight

		KRelax(dest, index, nodeWeight, devDistances, Queue, founds, level, devPaths);
	}
else {*/
	bool flag = true;
	edge_t k = start + (threadIdx.x & MOD2<VW_SIZE>::value);
	while (flag) {
		while (k < end && founds < REG_LIMIT) {
			const auto dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdges + k);		// X: edge index	Y: edge weight

			//devPaths[dest] = index;

			KRelax(dest, index, nodeWeight, devDistances, Queue, founds, level, d_distances);
			k += VW_SIZE;
		}
		if (__any(founds >= REG_LIMIT)) {
            int prefix_sum = founds;
            const int warp_offset = WarpExclusiveScan<>::AddAtom(prefix_sum, devF2SizePrt);
            thread::RegToGlobal(Queue, founds, devF2 + warp_offset + prefix_sum);
			founds = 0;
		} else
			flag = false;
	}
//}
}

//------------------------------------------------------------------------------

/*__global__ void DynamicKernel( 	EdgeT* __restrict__ devEdge,
								hdist_t* __restrict__ devDistance,
                                node_t* devF2,
								const node_t index,
                                const weight_t nodeWeight,
                                const edge_t start,
                                const edge_t end,
								const int level,
int *devPaths) {

	int* devF2SizePrt = &devF2Size[level & 3];
	const int ID = blockIdx.x * BLOCKDIM + threadIdx.x;

	node_t Queue[REG_LIMIT];
	int founds = 0;
	for (int k = start + ID; k < end; k += gridDim.x * BLOCKDIM) {
		const auto dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdge + k);		// X: edge index	Y: edge weight

		KRelax(dest, index, nodeWeight, devDistance, Queue, founds, level, devPaths);
	}
    int prefix_sum = founds;
    const int warp_offset = WarpExclusiveScan<>::AddAtom(prefix_sum, devF2SizePrt);
    thread::RegToGlobal(Queue, founds, devF2 + warp_offset + prefix_sum);
}
*/

template<int VW_SIZE>
__device__ __forceinline__ void DynamicParallelism( EdgeT* __restrict__ devEdges,
                                                    hdist_t* __restrict__ devDistances,
                                                    node_t* devF2,
													const node_t index,
                                                    const weight_t nodeWeight,
                                                    const edge_t start,
                                                    edge_t& end,
													const int level,
int* __restrict__ devPaths) {
/*
	if (DYNAMIC_PARALLELISM) {
		const int degree = end - start;
		if (degree >= THRESHOLD_G) {
			if ((threadIdx.x & MOD2<VW_SIZE>::value) == 0) {
				const int DynGridDim = (degree + (BLOCKDIM * EDGE_PER_TH) - 1) >> (LOG2<BLOCKDIM>::value + LOG2<EDGE_PER_TH>::value);
				DynamicKernel<<< DynGridDim, BLOCKDIM>>> (devEdges, devDistances, devF2, index, nodeWeight, start, end, level, devPaths);
			}
			end = INT_MIN;
		}
	}
*/
}


} //@kernels
