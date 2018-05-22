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
#include "../../include/Device/HBFGraph.cuh"
#include "Graph.hpp"
#include "XLib.hpp"

using namespace graph;

__device__ int devF2Size[4];
//__constant__ int *devPaths;
/*
struct __align__(8) EdgeT {
    int x;
    float y;
};*/

HBFGraph::HBFGraph(curnet::base_graph& graph,
                   const bool _inverse_graph,
                   const int _degree_options) :
                   cudaGraphWeight(graph, _inverse_graph, _degree_options)
                   {

	//cudaMalloc(&devDistances, graph.V * sizeof (hdist_t));
    const int delta = (1 << 20) * 16;   //16 MB free

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    cudaMalloc(&devF1, (free - delta) / 2 );
    cudaMalloc(&devF2, (free - delta) / 2);
    __CUDA_ERROR("HBFGraph Allocation");

    max_frontier_size = ((free - delta) / 2) / sizeof(node_t);
    StreamModifier::thousandSep();
    //std::cout << "Max frontier size: " <<  max_frontier_size << std::endl;
    StreamModifier::resetSep();

    devDistanceInit = new hdist_t[graph.v()];
    std::fill(devDistanceInit, devDistanceInit + graph.v(), INF);

    //cudaMalloc(&devPaths, graph.V * sizeof(int));
    cudaMalloc(&d_distances, graph.v() * sizeof(long long unsigned));
}

HBFGraph::~HBFGraph() {
    cudaFree(devF1);
    cudaFree(devF2);
    delete[] devDistanceInit;
}

static const float INF_FLOAT = std::numeric_limits<float>::infinity();



__global__ void initKernel(long long unsigned* d_distances, int source, int V) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < V; i += stride) {
        //DistPath tmp = { INF_FLOAT, -1 };
        DistPath tmp = { -1, INF_FLOAT };
        d_distances[i] = reinterpret_cast<long long unsigned&>(tmp);
    }
    /*for (int i = id; i < num_sources; i += stride) {
        auto src = sources[i];

        d_distances[src] = reinterpret_cast<long long unsigned&>(tmp);
    }*/
    if (id == 0) {
        //DistPath tmp = { 0, -1 };
        DistPath tmp = { -1, 0.0f };
        d_distances[source] = reinterpret_cast<long long unsigned&>(tmp);
    }
}

/*
void HBFGraph::init(const node_t Sources[], int nof_sources) {
	cudaMemcpy(devF1, Sources, nof_sources * sizeof(node_t), cudaMemcpyHostToDevice);

    int* tmp_source;
    cudaMalloc(&tmp_source, nof_sources * sizeof(int));
    cudaMemcpy(tmp_source, Sources, nof_sources * sizeof(int), cudaMemcpyHostToDevice);
    __CUDA_ERROR("BellmanFord Kernel Init");

    initKernel<<< (graph.V + 255) / 256, 256>>>
        (d_distances, tmp_source, nof_sources, graph.V);

    cudaFree(tmp_source);

	/*cudaMemcpy(devDistances, devDistanceInit, graph.V * sizeof (hdist_t), cudaMemcpyHostToDevice);
	if (nof_sources == 1){
        const hdist_t zero = ZERO;
		cudaMemcpy(devDistances + Sources[0], &zero, sizeof(hdist_t), cudaMemcpyHostToDevice);
	}*/
    /*int SizeArray[4] = {0, 0, 0, 0};
	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);
	//cudaMemset(devPaths, 0xFF, graph.V * sizeof(int));
    __CUDA_ERROR("BellmanFord Kernel Init");

    global_sync::Reset();
}*/

__global__ void initKernel2(long long unsigned* d_distances, int source, int V) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < V; i += stride) {
        DistLevel tmp = { INT_MAX, INF_FLOAT };
        d_distances[i] = reinterpret_cast<long long unsigned&>(tmp);
    }
    if (id == 0) {
        DistLevel tmp = { 0, 0.0f };
        d_distances[source] = reinterpret_cast<long long unsigned&>(tmp);
    }
}


void HBFGraph::init(node_t source, bool enable_path) {
	cudaMemcpy(devF1, &source, sizeof(node_t), cudaMemcpyHostToDevice);

    if (!enable_path) {
        initKernel2<<< (graph.v() + 255) / 256, 256>>>
            (d_distances, source, graph.v());
    }
    else {
        initKernel<<< (graph.v() + 255) / 256, 256>>>
            (d_distances, source, graph.v());
    }
    int SizeArray[4] = {0, 0, 0, 0};
	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);
    __CUDA_ERROR("BellmanFord Kernel Init");
}
