#include "cuModuleSCC.cuh"

#include "device/cudaGraph.cuh"
#include "../timer.h"

#include <vector>
#include "cudaUtil.cuh"

using namespace scc4k;
void cu_scc(
	curnet::base_graph &graph, 
	std::vector<int> &colors){

//#define SCC_CPU
#ifdef SCC_CPU
//	free(graph.countSCC());
//	graph.copyColorToVector(colors);
#else
	scc4k::cudaGraph devGraph(graph);
	devGraph.Reset();

	int V = graph.v();


	bool executionOK = devGraph.cudaSCC4K(16, // MIN_WARP BFS
										32,  // MAX_WARP_BFS
										false, //dynamic parallellism
										0, //duplicate remove
										true, //euristic
										0, //warp size fwbw. With "0", it enables BFS-4K, otherwise quadratic
										4, //warp size coloring
										true, //color direction forward
										-1, //max trim iteration. -1 = FULL
										3, //fwbw iterations
										-1 //id run. Not used, legacy code
										);//fixed parameters ATM!*/
	devGraph.copyColorToVector(colors);
	/*
	std::vector<int> scc = graph.scc();
	for(int i = 0; i < V; ++i)
	{
		if(scc[i] != scc[colors[i]-1]) std::cout << "Error at " << i << ": expected " << scc[i] << ", got " << colors[i] << std::endl;
	}
	*/
#endif
}


void cu_bfs(
	curnet::base_graph *graph, 
	std::vector<std::vector<float>> &distance,
	std::vector<int> source){
	
	//TIMEHANDLE start;
	//double ptime = 0.0;

	//start = start_time();
	scc4k::cudaGraph devGraph(*graph);
	//ptime = end_time(start);
	//std::cout << "Creation graph: " << ptime << " ms" << std::endl;
	
	//start = start_time();
	for(int i = 0; i < source.size(); i++)
	{
		int s = source[i];
		//TIMEHANDLE start = start_time();
		//double ptime = end_time(start);
		//std::cout << "copy time int: " << ptime << std::endl;
		//devGraph.Reset(source.data(), source.size());
		devGraph.Reset(&s, 1);
		std::vector<float> vv(graph->v());
		devGraph.copyDistanceToVector(vv);

		int V = graph->v();
	
		bool executionOK = devGraph.cudaBFS4K(16, // MIN_WARP BFS
											32,  // MAX_WARP_BFS
											false, //dynamic parallellism
											0, //duplicate remove
											true, //euristic
											0, //warp size fwbw. With "0", it enables BFS-4K, otherwise quadratic
											1  // number of sources
											);//fixed parameters ATM!
	
		devGraph.copyDistanceToVector(distance[i]);
	}
	
	//ptime = end_time(start);
	//std::cout << "Visiting " << source.size() << " vertices: " << ptime << " ms" << std::endl;
	/*
	std::vector<int> dd = graph->bfs(source);
	
	for(int i = 0; i < V; ++i)
	{
		if(dd[i] != distance[i]) std::cout << "Error at " << i << ": expected " << dd[i] << ", got " << distance[i] << std::endl;
	}
	*/
}
