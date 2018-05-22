#include "cudaGraph.cuh"

#include "../../../timer.h"

namespace scc4k{

	__device__ int duplicateCounter;
	__device__ int devF2Size[4];


	cudaGraph::cudaGraph(curnet::base_graph& _graph) : graph(_graph){
		cudaError("At constructor!");
		V = graph.v();
		E = graph.e();
	
	TIMEHANDLE start = start_time();
	//#define USE_SINGLE_CUDAMALLOC
	#ifdef USE_SINGLE_CUDAMALLOC
	  //globalPointer
	  size_t globalSize = 
	  (V + 1) * sizeof (int) +
		E * sizeof (int) +
		(V + 1) * sizeof (int) +
		E * sizeof (int) +
		V * sizeof (dist_t) +
		V*2 * sizeof (color_t) +
		V * sizeof (uint8_t) +
		(V*3+1) * sizeof (int) +
		(V*3+1) * sizeof (int) +
		V * sizeof(dist_t) +
		sizeof(int) +
		sizeof(int) +
		sizeof(int);
	
		// --------------- Frontier Allocation -------------------
		cudaError("Before call meminfo");
		size_t free, total;
		cudaMemGetInfo(&free, &total);
	
		cudaError("After call meminfo");
		size_t frontierSize = ((free - globalSize) / 8) - 2* 500 * 1024;
	 
		//std::cout << "Frontier size: " <<  frontierSize;
		//frontierSize = min(static_cast<int>((1<<27 - globalSize) >> 1), static_cast<int>(frontierSize)); //empirical limit value before crushing malloc performance!
		//std::cout << " - " <<  frontierSize << std::endl;
	 
		globalSize += frontierSize + frontierSize;
	
		//std::cout << globalSize << std::endl;

		allocFrontierSize = frontierSize / sizeof(int);

		cudaError("Graph Frontier Allocation");
	
		size_t gS[15] = { (V + 1) * sizeof (int), E * sizeof (int), (V + 1) * sizeof (int), E * sizeof (int), V * sizeof (dist_t), V*2 * sizeof (color_t), V * sizeof (uint8_t), 
						(V*3+1) * sizeof (int),	(V*3+1) * sizeof (int),	V * sizeof(dist_t), sizeof(int), sizeof(int), sizeof(int), frontierSize, frontierSize};
		size_t gSC[15];
		gSC[0] = 0;
		for(int i = 1; i < 15; ++i) { gSC[i] = gSC[i-1] + gS[i-1]; }
	 
		cudaError("before cudaMemcpy");
		cudaMalloc((void **) &globalPointer, globalSize);
		cudaError("cudaMemcpy global");
		devOutNodes = (int*) (globalPointer + gSC[0]);
		devOutEdges = (int*) (globalPointer + gSC[1]);
		devInNodes = (int*) (globalPointer + gSC[2]);
		devInEdges = (int*) (globalPointer  + gSC[3]);

		devDistance = (dist_t*) (globalPointer + gSC[4]);
		devColor = (int*) (globalPointer + gSC[5]);
		devState = (uint8_t*) (globalPointer + gSC[6]);
		devPivotPerColor = (int*) (globalPointer + gSC[7]);
		devPivotPerColorSize = (int*) (globalPointer +  + gSC[8]);
		devDistanceIn = (dist_t*) (globalPointer + gSC[9]);
		devGlobalCounter = (int*) (globalPointer + gSC[10]);
	
		D_hasToContinue = (int*) (globalPointer + gSC[11]);
		D_num_of_scc = (int*) (globalPointer + gSC[12]);
	
		devF1 = (int*)  (globalPointer + gSC[13]);
		devF2 = (int*)  (globalPointer + gSC[14]);
	#else
		cudaError("Before all malloc");
		cudaMalloc((void **) &devOutNodes, (V + 1) * sizeof (int));
		cudaMalloc((void **) &devOutEdges, E * sizeof (int));
		cudaMalloc((void **) &devInNodes, (V + 1) * sizeof (int));
		cudaMalloc((void **) &devInEdges, E * sizeof (int));
		//cudaMalloc(&devF1, V * (F_MUL) * sizeof (int));
		//cudaMalloc(&devF2, V * (F_MUL) * sizeof (int));
		cudaMalloc((void **) &devDistance, V * sizeof (dist_t));

		cudaMalloc((void **) &devColor, V*2 * sizeof (color_t));
		cudaMalloc((void **) &devState, V * sizeof (uint8_t));
		cudaMalloc((void **) &devPivotPerColor, (V*3+1) * sizeof (int));
		cudaMalloc((void **) &devPivotPerColorSize, (V*3+1) * sizeof (int));
		cudaMalloc((void **) &devDistanceIn, V * sizeof(dist_t));

		cudaMalloc((void **) &devGlobalCounter, sizeof(int));
	
		cudaMalloc((void **) &D_hasToContinue, sizeof(int));
		cudaMalloc((void **) &D_num_of_scc, sizeof(int));
		cudaError("After all malloc");
	
		// --------------- Frontier Allocation -------------------

		size_t free, total;
		cudaMemGetInfo(&free, &total);
		size_t frontierSize = (free / 32) - 2* 500 * 1024;
		//frontierSize = min( frontierSize, static_cast<size_t>(1 << 10*10*7) ); //max 128 MB allocation
	 
		//std::cout << "Frontier size: " <<  frontierSize;
		//frontierSize = min(frontierSize, static_cast<size_t>(1 << 27));
		//std::cout << " - " <<  frontierSize << std::endl;
		cudaMalloc((void **) &devF1, frontierSize);
		cudaMalloc((void **) &devF2, frontierSize);

		allocFrontierSize = frontierSize / sizeof(int);

		cudaError("Graph Frontier Allocation");
	 #endif
		cudaMemcpy((void**) devOutNodes, graph.v_out(), (V + 1) * sizeof (int), cudaMemcpyHostToDevice);
		cudaMemcpy((void**) devOutEdges, graph.e_out(), E * sizeof (int), cudaMemcpyHostToDevice);
		cudaError("cudaMemcpy");

		double ptime = end_time(start);
//		std::cout << "Construction: " << ptime << std::endl;	

		cudaMemcpy((void**) devInNodes, graph.v_in(), (V + 1) * sizeof (int), cudaMemcpyHostToDevice);
		cudaMemcpy((void**) devInEdges, graph.e_in(), E * sizeof (int), cudaMemcpyHostToDevice);

		frontier = new int[V];

		const int ZERO = 0;
		cudaMemcpyToSymbol(duplicateCounter, &ZERO, sizeof (int));

		cudaError("Graph Allocation");

	}


	cudaGraph::~cudaGraph()
	{
		cudaError("Init Graph Deallocation");
		TIMEHANDLE t1 = start_time();	
		delete[] frontier;
	#ifdef USE_SINGLE_CUDAMALLOC
		cudaFree(globalPointer);
		cudaError("free global pointer");
	#else
		cudaError("Init Graph Deallocation");
		cudaFree(devOutNodes);
		cudaFree(devOutEdges);
		cudaFree(devInNodes);
		cudaFree(devInEdges);
		cudaFree(devDistance);
		cudaError("Mid");
		cudaFree(devColor);
		cudaFree(devState);
		cudaFree(devPivotPerColor);
		cudaFree(devPivotPerColorSize);
		cudaFree(devDistanceIn);

		cudaFree(devGlobalCounter);

		cudaFree(D_hasToContinue);
		cudaFree(D_num_of_scc);

		cudaFree(devF1);
		cudaFree(devF2);
		cudaError("Graph Deallocation");
	#endif
		double ptime = end_time(t1);
//		std::cout << "Destuction: " << ptime << std::endl;
	}

	bool cudaGraph::copyColorToVector(std::vector<int> &c)
	{
		int *colors = new int[V*2]();
		cudaMemcpy(colors, (void*) (devColor), (2*V) * sizeof (color_t), cudaMemcpyDeviceToHost);
		c.clear();
		for(int i = 0; i < V*2; i+=2) c.push_back(-colors[i]);

		delete[] colors;
		return true;
	}
	bool cudaGraph::copyDistanceToVector(dist_t* v)
	{
		cudaError("Init copy distance");
		cudaMemcpy(v, (void*) (devDistance), (V) * sizeof (dist_t), cudaMemcpyDeviceToHost);
		cudaError("End copy distance");
		return true;
	}
	bool cudaGraph::copyDistanceToVector(std::vector<float> &v)
	{
		cudaError("Init copy distance");
		dist_t* vv = new dist_t[V];
		cudaMemcpy(vv, (void*) (devDistance), (V) * sizeof (dist_t), cudaMemcpyDeviceToHost);
		for(int i = 0; i < V; i++)
		{
			//if(i < 100) std::cout << vv[i] << ", " << INF << std::endl;
			
			if(vv[i] == INF)
				v[i] = std::numeric_limits<float>::infinity();
			else
			{
				//std::cout << "No inf!" << std::endl;
				v[i] = static_cast<float>(vv[i]);
				//std::cout << v[i] << std::endl;
			}
			//v[i] = static_cast<float>(vv[i]);
				
		}
		delete[] vv;
		cudaError("End copy distance");
		return true;
	}
}

#include "WorkEfficientKernel/BFS_WE_Kernels1.cu"
#include "WorkEfficientKernel/BFS_WE_Dynamic.cu"

// ----------------------- GLOBAL SYNCHRONIZATION --------------------------------

#define __GLOBAL_DEVICE__ __global__
#define NAME1 BFS_KernelMainGLOB
#define NAME1B BFS_KernelMainGLOBB

#include "WorkEfficientKernel/BFS_WE_KernelMain.cu"

#undef __GLOBAL_DEVICE__
#undef NAME1
#undef NAME1B

#define __GLOBAL_DEVICE__ __device__
#define NAME1 BFS_KernelMainDEV
#define NAME1B BFS_KernelMainDEVB

#include "WorkEfficientKernel/BFS_WE_KernelMain.cu"

#undef __GLOBAL_DEVICE__
#undef NAME1
#undef NAME1B

// ----------------------------------------------------------------------------------

//#include "WorkEfficientKernel/BFS_WE_KernelDispatch.cu"
#include "Util/GlobalSync.cu"
//#include "WorkEfficientKernel/BFS_WE_Block.cu"

#include "BFS_WorkEfficient.cu"
