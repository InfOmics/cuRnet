#pragma once

#include <base_graph/base_graph.hpp>
#include <sstream>
#include <iostream>
#include "fUtil.h"
#include "cudaUtil.cuh"
//#include "Timer.cuh"
#include "printExt.h"
#include "../../config.h"
#include "definition.cuh"

#include <deque>
#include <vector>

namespace scc4k{

enum Method{
	Trim,ForwardClosure, BackwardClosure, PivotSelection, CopyColors, CopyColorsForward, CopyColorsBackward, SCC_Found,
	ParSetColorToNodeEnum, ParSetColorPivotEnum, FwdMaxColorEnum, BackwardClosureColoring,
	ForwardClosureNodes, BackwardClosureNodes, TotalVertexSCC, TotalNumOfSCC,
	TRIM, FWBW, COLORING
};
typedef struct LogValues{
	int recursionLevel;
	int subRecursionLevel;
	Method method;
	float time;
	int   size;
} LogValues;

class cudaGraph {
		curnet::base_graph& graph;
   void* globalPointer;
		int* devOutNodes, *devOutEdges, *devOutDegree;
		int* devInNodes, *devInEdges, *devInDegree;
		int *devF1, *devF2;
		dist_t* devDistance;
		int V, E;
		int allocFrontierSize;

		// aggiunta per SCC
		color_t* devColor;
		int* devPivotPerColor;
		int* devPivotPerColorSize;
		uint8_t* devState;
		dist_t* devDistanceIn;

		int* devGlobalCounter;

		int* frontier;

		const int MAX_TIME_COMPUTE = 7000;
		bool timeout;

		int* D_hasToContinue;
		int* D_num_of_scc;

		bool globalErr = false;
		int recursionLevel;

		const int WARP_SIZE_MAGIC_VALUE = 1024;
		
		
		// Template definition CUDA BFS4k		
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE>
		bool cudaBFS4K(int source);
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC>
		bool cudaBFS4K(int WARP_SIZE, int source);
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE>
		bool cudaBFS4K(bool EURISTIC, int WARP_SIZE, int source);
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM>
		bool cudaBFS4K(int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE, int source);
		template<int MIN_WARP, int MAX_WARP>
		bool cudaBFS4K(bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE, int source);
		template<int MIN_WARP>
		bool cudaBFS4K(int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE, int source);
		
		// Template definition CUDA SCC4k
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD>
		bool cudaSCC4K(int maxTrimIteration, int totalFWBWIteration, int totalRun);
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL>
		bool cudaSCC4K(bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int totalRun);
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW>
		bool cudaSCC4K(int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int totalRun);
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC>
		bool cudaSCC4K(int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int totalRun);
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE>
		bool cudaSCC4K(bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int totalRun);
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM>
		bool cudaSCC4K(int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int totalRun);
		template<int MIN_WARP, int MAX_WARP>
		bool cudaSCC4K(bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int totalRun);
		template<int MIN_WARP>
		bool cudaSCC4K(int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int totalRun);

	public:
		cudaGraph(curnet::base_graph& graph);
		//cudaGraph() : graph(){};
		~cudaGraph();

		void Reset(const int Sources[], int nof_sources = 1);
		void Reset();

		inline bool cudaBFS4K(int* source);
		bool cudaBFS4K_N(int nof_tests = 1);

		bool cudaSCC4K(int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int idRun);
		bool cudaBFS4K(int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE, int source);
		bool copyColorToVector(std::vector<int>& c);
		bool copyDistanceToVector(unsigned * d);
		bool copyDistanceToVector(std::vector<float>& c);


		template<bool forward, VisitType visitType, int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, int WARP_SIZEE>
		inline int closure(int* devNodes, int* devEdges, dist_t* devDistance, int FrontierSize);

		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, int WARP_SIZE_COL, bool COLOR_FWD>
		inline int coloring(std::deque<LogValues>& loggedValues, LogValues& value_to_log);
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW>
		inline int fwbw(std::deque<LogValues>& loggedValues, LogValues& value_to_log);
		inline int trim1(std::deque<LogValues>& loggedValues, LogValues& value_to_log);

	private:
		inline void FrontierDebug(int F2Size, int level, bool DEBUG = false);

		template<int MIN_VALUE, int MAX_VALUE>
		inline int logValueHost(int Value);
};

}
