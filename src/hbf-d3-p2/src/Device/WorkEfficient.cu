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
#include "Device/HBFGraph.cuh"
#include "XLib.hpp"

#include "Kernels/WorkEfficient_KernelDispath.cuh"

#include "Kernels/WorkEfficientKernel2.cuh"

void HBFGraph::WorkEfficient_PCSF(int source, DistPath* dists_paths, bool enable_path) {
    int SizeArray[4];
    int maxFrontier = std::numeric_limits<int>::min();
    std::vector<int> host_frontiers;

	int level = 1, F1Size = 1, F2Size;
	this->init(source, enable_path);

	do {
        if (enable_path)
            DynamicVirtualWarp1(F1Size, level);
        else
            DynamicVirtualWarp2(F1Size, level);

		cudaMemcpyFromSymbol(SizeArray, devF2Size, sizeof(int) * 4);

		F2Size = SizeArray[level & 3];
		F1Size = F2Size;
        //std::cout << "level: "<< level << " " << F2Size << std::endl;
		level++;
		FrontierDebug(F2Size, level);
		std::swap<int*>(devF1, devF2);
		maxFrontier = std::max(maxFrontier, F2Size);
	} while ( F2Size > 0 );

	cudaMemcpy(dists_paths, d_distances, graph.v() * sizeof(long long unsigned),
               cudaMemcpyDeviceToHost);
}






void HBFGraph::FrontierDebug(const int FSize, const int level) {
    if (FSize > max_frontier_size)
        __ERROR("Device memory not sufficient to contain the vertices frontier");
	if (CUDA_DEBUG) {
		__CUDA_ERROR("BellmanFord Host");

		std::cout << "level: " << level << "\tF2Size: " << FSize << std::endl;
		if (CUDA_DEBUG >= 2) {
            node_t* tmpF1 = new node_t[graph.v()];
			cudaMemcpy(tmpF1, devF1, FSize * sizeof(node_t), cudaMemcpyDeviceToHost);
			printExt::device::printArray(tmpF1, FSize, "Frontier:\t");
		}
	}
}

inline void HBFGraph::DynamicVirtualWarp1(const int F1Size, const int level) {
    int size = numeric::log2(RESIDENT_THREADS / F1Size);
	if (MIN_VW >= 1 && size < LOG2<MIN_VW>::value)
		size = LOG2<MIN_VW>::value;
	if (MAX_VW >= 1 && size > LOG2<MAX_VW>::value)
		size = LOG2<MAX_VW>::value;

    #define fun(a)  kernels::BF_Kernel1<(a), false>                             \
                        <<< _Div(graph.v(), (BLOCKDIM / (a)) * ITEM_PER_WARP),    \
                        BLOCKDIM,                                               \
                        SMem_Per_Block<char, BLOCKDIM>::value >>>               \
                        (devOutNodes, devOutEdges, devDistances, devF1, devF2,  F1Size, level, d_distances);

    def_SWITCH(size);

    #undef fun
}

inline void HBFGraph::DynamicVirtualWarp2(const int F1Size, const int level) {
    int size = numeric::log2(RESIDENT_THREADS / F1Size);
	if (MIN_VW >= 1 && size < LOG2<MIN_VW>::value)
		size = LOG2<MIN_VW>::value;
	if (MAX_VW >= 1 && size > LOG2<MAX_VW>::value)
		size = LOG2<MAX_VW>::value;

    #define fun(a)  kernels::BF_Kernel2<(a), false>                             \
                        <<< _Div(graph.v(), (BLOCKDIM / (a)) * ITEM_PER_WARP),    \
                        BLOCKDIM,                                               \
                        SMem_Per_Block<char, BLOCKDIM>::value >>>               \
                        (devOutNodes, devOutEdges, devDistances, devF1, devF2,  F1Size, level, d_distances);

    def_SWITCH(size);

    #undef fun
}
