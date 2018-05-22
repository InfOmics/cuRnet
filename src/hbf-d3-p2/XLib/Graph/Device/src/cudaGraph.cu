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
#include "../include/cudaGraphNamespace.cuh"
#include "../include/cudaGraph.cuh"

namespace cuda_graph {

cudaGraph::cudaGraph(GraphSTD& _graph,
                     const bool _inverse_graph,
                     const int _degree_options) :
                     graph(_graph),
                     inverse_graph(_inverse_graph),
                     degree_options(_degree_options) {

	cudaMalloc(&devOutNodes, (graph.V + 1) * sizeof(edge_t));
	cudaMalloc(&devOutEdges, graph.E * sizeof(node_t));

    if (inverse_graph) {
        cudaMalloc(&devInNodes, (graph.V + 1) * sizeof(edge_t));
        cudaMalloc(&devInEdges, graph.E * sizeof(node_t));
    }

	if ((degree_options & IN_DEGREE) && (degree_options & OUT_DEGREE))
		cudaMalloc(&devInOutDegrees, graph.V * sizeof(int2));
	else if (degree_options & IN_DEGREE)
		cudaMalloc(&devInDegrees, graph.V * sizeof(degree_t));
	else if (degree_options & OUT_DEGREE)
		cudaMalloc(&devOutDegrees, graph.V * sizeof(degree_t));

	__CUDA_ERROR("Graph Allocation");
}

cudaGraph::~cudaGraph() {
    cudaFree(devOutNodes);
    cudaFree(devOutEdges);

    if (inverse_graph) {
        cudaFree(devInNodes);
        cudaFree(devInEdges);
    }
    if ((degree_options & IN_DEGREE) && (degree_options & OUT_DEGREE))
		cudaFree(devInOutDegrees);
	else if (degree_options & IN_DEGREE)
		cudaFree(devInDegrees);
	else if (degree_options & OUT_DEGREE)
		cudaFree(devOutDegrees);
}

void cudaGraph::copyToDevice() {
	cudaMemcpy(devOutNodes, graph.OutNodes, (graph.V + 1) * sizeof(edge_t), cudaMemcpyHostToDevice);
	cudaMemcpy(devOutEdges, graph.OutEdges, graph.E * sizeof(node_t), cudaMemcpyHostToDevice);

    if (inverse_graph) {
        cudaMemcpy(devInNodes, graph.InNodes, (graph.V + 1) * sizeof(edge_t), cudaMemcpyHostToDevice);
        cudaMemcpy(devInEdges, graph.InEdges, graph.E * sizeof(node_t), cudaMemcpyHostToDevice);
    }

	if ((degree_options & IN_DEGREE) && (degree_options & OUT_DEGREE)) {
		int2* tmpInOutDegrees = new int2[graph.V];
		for (int i = 0; i < graph.V; i++)
			tmpInOutDegrees[i] = make_int2(graph.InDegrees[i], graph.OutDegrees[i]);

		cudaMemcpy(devInOutDegrees, tmpInOutDegrees, graph.V * sizeof(int2), cudaMemcpyHostToDevice);
		delete[] tmpInOutDegrees;
	}
    else if (degree_options & IN_DEGREE)
		cudaMemcpy(devInDegrees, graph.InDegrees, graph.V * sizeof(degree_t), cudaMemcpyHostToDevice);
	else if (degree_options & OUT_DEGREE)
		cudaMemcpy(devOutDegrees, graph.OutDegrees, graph.V * sizeof(degree_t), cudaMemcpyHostToDevice);

	__CUDA_ERROR("Graph Copy To Device");
}

} //@cuda_graph
