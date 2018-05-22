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

#include "Device/WorkEfficient_KernelFun2.cuh"

namespace kernels {

template<int VW_SIZE, bool SYNC>
__global__ void BF_Kernel2 (	edge_t* __restrict__  devNodes,
								EdgeT* __restrict__    devEdges,
								hdist_t* __restrict__ devDistances,
								node_t* __restrict__  devF1,
								node_t* __restrict__  devF2,
								const int devF1Size,
                                const dist_t level,
				                long long unsigned* __restrict__ d_distances) {

	node_t* devF2SizePrt = &devF2Size[level & 3];	// mod 4
	if (blockIdx.x == 0 && threadIdx.x == 0)
		devF2Size[(level + 1) & 3] = 0;

	int founds = 0;
	node_t Queue[REG_LIMIT];

	const int VirtualID = (blockIdx.x * BLOCKDIM + threadIdx.x) / VW_SIZE;
	const int Stride = gridDim.x * (BLOCKDIM / VW_SIZE);

	const int size = ceilf(__fdividef(devF1Size, gridDim.x));
	const int maxLoop = (size + BLOCKDIM / VW_SIZE - 1) >> (LOG2<BLOCKDIM>::value - LOG2<VW_SIZE>::value);

	for (int t = VirtualID, loop = 0; loop < maxLoop; t += Stride, loop++) {
        node_t index;
		weight_t nodeWeight;
        edge_t start, end = INT_MIN;
		if (t < devF1Size) {
	        index = devF1[t];
			nodeWeight = reinterpret_cast<DistPath*>(d_distances)[index].dist;
			start = devNodes[index];
			end = devNodes[index + 1];
		}
		EdgeVisit2<VW_SIZE>(devEdges, devDistances, devF2, devF2SizePrt, index, nodeWeight, start, end, Queue, founds, level, d_distances);
	}

    int prefix_sum = founds;
    const int warp_offset = WarpExclusiveScan<>::AddAtom(prefix_sum, devF2SizePrt);
    thread::RegToGlobal(Queue, founds, devF2 + warp_offset + prefix_sum);
}

} //@kernels
