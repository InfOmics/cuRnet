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
#include "../../include/Device/DegreeMarker.cuh"
#include "../../include/Device/HBFGraph.cuh"
#include "XLib.hpp"
using namespace timer_cuda;

namespace degree_marker {

__global__ void InEdgeMarker(int2* __restrict__ devOutEdges,
                             const int E,
                             const int* __restrict__ devInDegree) {

	const int GlobalID = blockIdx.x * BLOCKDIM + threadIdx.x;

	for (int i = GlobalID; i < E; i += gridDim.x * BLOCKDIM) {
		const int2 dest = devOutEdges[i];
		if (devInDegree[dest.x] == 1)
			devOutEdges[i] = make_int2(dest.x | 0x80000000, dest.y);
	}
}

template<int UNDIRECTED_T>
__global__ void OutEdgeMarker(int2* __restrict__ devOutEdges,
                              const int E,
                              const int* __restrict__ devOutDegrees) {

	const int GlobalID = blockIdx.x * BLOCKDIM + threadIdx.x;

	for (int i = GlobalID; i < E; i += gridDim.x * BLOCKDIM) {
		const int2 dest = devOutEdges[i];
		if (devOutDegrees[dest.x] == UNDIRECTED_T)
			devOutEdges[i] = make_int2(dest.x | 0x40000000, dest.y);
	}
}

template<int UNDIRECTED_T>
__global__ void InOutEdgeMarker(int2* __restrict__ devOutEdges,
                                const int E,
                                const int2* __restrict__ devInOutDegrees) {

	const int GlobalID = blockIdx.x * BLOCKDIM + threadIdx.x;

	for (int i = GlobalID; i < E; i += gridDim.x * BLOCKDIM) {
		const int2 dest = devOutEdges[i];
		const int2 InOutDegree = devInOutDegrees[dest.x];
		if (InOutDegree.x == 1 && InOutDegree.y == UNDIRECTED_T)
			devOutEdges[i] = make_int2(dest.x | 0xC0000000, dest.y);
		else if (InOutDegree.x == 1)
			devOutEdges[i] = make_int2(dest.x | 0x80000000, dest.y);
		else if (InOutDegree.x == UNDIRECTED_T)
			devOutEdges[i] = make_int2(dest.x | 0x40000000, dest.y);
	}
}

} //@degree_marker

// ---------------------------------------------------------------------------------------------

using namespace degree_marker;
/*
void HBFGraph::markDegree() {
	if (IN_DEGREE_OPT || OUT_DEGREE_OPT) {
		Timer<DEVICE> TM;
        TM.start();

		if (IN_DEGREE_OPT && OUT_DEGREE_OPT) {
            if (graph.Direction == EdgeType::UNDIRECTED)
			     InOutEdgeMarker<1> <<<_Div(graph.E, BLOCKDIM * 8), BLOCKDIM>>> (devOutEdges, graph.E, devInOutDegrees);
            else
                 InOutEdgeMarker<0> <<<_Div(graph.E, BLOCKDIM * 8), BLOCKDIM>>> (devOutEdges, graph.E, devInOutDegrees);
        }
		else if (IN_DEGREE_OPT)
			InEdgeMarker <<<_Div(graph.E, BLOCKDIM * 8), BLOCKDIM>>> (devOutEdges, graph.E, devInDegrees);
		else if (OUT_DEGREE_OPT) {
            if (graph.Direction == EdgeType::UNDIRECTED)
			     OutEdgeMarker<1> <<<_Div(graph.E, BLOCKDIM * 8), BLOCKDIM>>> (devOutEdges, graph.E, devOutDegrees);
            else
                OutEdgeMarker<0> <<<_Div(graph.E, BLOCKDIM * 8), BLOCKDIM>>> (devOutEdges, graph.E, devOutDegrees);
        }
		//else if (OUT_DEGREE_OPT == 2 && UNDIRECTED_T)
		//	OutVertexMarkerExt2 <<<DIV(V, BLOCKDIM * 32), 256>>> (devOutNode, devOutEdges, V, devOutDegrees);

		TM.getTime("Degree Kernel Time");
		__CUDA_ERROR("Degree Marker");
	}
}*/
