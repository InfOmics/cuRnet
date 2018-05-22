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

__device__ __forceinline__ void KRelax2(	const EdgeT dest,
                                        const node_t index,
                                        const weight_t node_weight,
					                    hdist_t* devDistance,
                                        node_t* Queue,
                                        int& founds,
                                        const int level,
					                    long long unsigned* d_distances) {
    using ull_t = unsigned long long;
    weight_t newWeight = node_weight + dest.y;

    //printf("(%d %d)   w: %f %f\n", index, dest.x, node_weight, newWeight);

    DistLevel toWrite = { level, newWeight };
	ull_t aa = atomicMin(reinterpret_cast<ull_t*>(d_distances + dest.x),
						 reinterpret_cast<ull_t&>(toWrite));
	DistLevel toTest = reinterpret_cast<DistLevel&>( aa );
    //printf("l: %d \t nl: %d \t  d: %f \t nw: %f\n",
    //        level, toTest.level, toTest.dist, newWeight);

	if (toTest.level != level && toTest.dist > newWeight)
		Queue[founds++] = dest.x;
}

using namespace data_movement::dynamic;

template<int VW_SIZE, int SIZE>
__device__ __forceinline__ void EdgeVisit2(	EdgeT* __restrict__ devEdges,
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

	bool flag = true;
	edge_t k = start + (threadIdx.x & MOD2<VW_SIZE>::value);
	while (flag) {
		while (k < end && founds < REG_LIMIT) {
			const auto dest = cub::ThreadLoad<cub::LOAD_LDG>(devEdges + k);		// X: edge index	Y: edge weight

			KRelax2(dest, index, nodeWeight, devDistances, Queue, founds, level, d_distances);
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
}

} //@kernels
