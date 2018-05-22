#pragma once
#include <deque>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sys/stat.h>

namespace scc4k{

long long int totalFrontierNodes;
long long int totalFrontierEdges;

#ifndef CUDA_ERROR
//#define CUDA_ERROR(s) cudaError(s)
#define CUDA_ERROR(s)
#endif


__host__ __device__ inline bool isForwardVisited(uint8_t tags) { return ( tags & 1); }
__host__ __device__ inline bool isForwardPropagate(uint8_t tags ) { return (tags & 4); }
__host__ __device__ inline bool isBackwardVisited(uint8_t tags) { return (tags & 2); }
__host__ __device__ inline bool isBackwardPropagate(uint8_t tags) { return ( tags & 8); }
__host__ __device__ inline void setForwardVisitedBit(uint8_t *tags) { *tags = ( *tags | 1); };
__host__ __device__ inline void setForwardPropagateBit(uint8_t *tags) { *tags = ( *tags | 4); };
__host__ __device__ inline void setBackwardVisitedBit(uint8_t *tags) { *tags = ( *tags | 2); };
__host__ __device__ inline void setBackwardPropagateBit(uint8_t *tags) { *tags = ( *tags | 8); };
__host__ __device__ inline void clearForwardVisitedBit(uint8_t *tags) { *tags = (*tags & ~1); };
__host__ __device__ inline void clearForwardPropagateBit(uint8_t *tags) { *tags = (*tags & ~4); };
__host__ __device__ inline void clearBackwardVisitedBit(uint8_t *tags) { *tags = (*tags & ~2); };
__host__ __device__ inline void clearBackwardPropagateBit(uint8_t *tags) { *tags = (*tags & ~8); };
__host__ __device__ inline bool isRangeSet(uint8_t tags) { return ( tags & 16); }
__host__ __device__ inline void rangeSet(uint8_t *tags) { *tags = ( *tags | 16); };
__host__ __device__ inline void setTrim1(uint8_t *tags) { *tags = ( *tags | 32); };
__host__ __device__ inline bool isTrim1(uint8_t tags) { return ( tags & 32); }
__host__ __device__ inline void setTrim2(uint8_t *tags) { *tags = ( *tags | 64); };
__host__ __device__ inline bool isTrim2(uint8_t tags) { return ( tags & 64); }
__host__ __device__ inline void setPivot(uint8_t *tags) { *tags = ( *tags | 128); };
__host__ __device__ inline bool isPivot(uint8_t tags) { return ( tags & 128); };

__global__ void updateColor(int V, uint8_t *State, color_t* color)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if(v >= V) return;
	int cIdx    = ( v << 1 );
	int cNewIdx = ( v << 1 ) | 1;
	color_t c = color[cIdx];

	if(c < 0) return;
	color_t newC = color[cNewIdx];

	if(newC != c)
	{
		bool needToUpdate = false;
		uint8_t s = 0;
		if(newC > 3*V)
		{
			//SCC
			c = -(newC - 3*V);
			needToUpdate = true;
			rangeSet(&s);
		}else if(newC > 2*V)
		{
			//B visit
			c = newC - 2*V;
			needToUpdate = true;
		}else if(newC > V)
		{
			//F visit
			c = newC - V;
			needToUpdate = true;
		}

		if(needToUpdate)
		{
			color[cIdx] = c;
			color[cNewIdx] = c;
			State[v] = s;
		}
	}
}

__global__ void updateColor(int V, color_t* color, uint8_t *State)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if(v >= V) return;
	int cIdx    = ( v << 1 );
	int cNewIdx = ( v << 1 ) | 1;
	uint8_t myState = State[v];
	if(isRangeSet(myState)) return;

	color_t newC = color[cNewIdx];

	if((myState & 3) > 0)
	{
		bool needToUpdate = false;
		uint8_t s = 0;
		if((myState & 3) == 3)
		{
			//SCC
			newC = -(newC);
			rangeSet(&s);
			needToUpdate = true;
		}else if((myState & 3) == 2)
		{
			//B visit
			newC = newC + 2*V;
			needToUpdate = true;
		}else if((myState & 3) == 1)
		{
			//F visit
			newC = newC + V;
			needToUpdate = true;
		}

		if(needToUpdate)
		{
			color[cIdx] = newC;
			color[cNewIdx] = newC;
			State[v] = s;
		}
	}
}

__global__ void ParSetFrontierColor(color_t* color, int* frontier, int FrontierSize, int V)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= FrontierSize) return;
	int v = frontier[idx];
	int cIdx    = ( v << 1 );
	int cNewIdx = ( v << 1 ) | 1;
	color_t c = -color[cNewIdx];
	color[cIdx]    = c;
	color[cNewIdx] = c;

}

__global__ void ParSetColor(int V, color_t* color, int* pivot)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	int c = color[cIdx];
	if(c < 0) return;
	// le SCC non sono considerate

	pivot[c] = v;
}

__global__ void ParSetColor(int V, color_t* color, int* pivot, uint8_t* State)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	int c = color[cIdx];
	if(c < 0) return;
	// le SCC non sono considerate

	pivot[c] = v;
}

__global__ void ParSetColor(int V, color_t* color, int* OutNodes, int* InNodes, int* pivot, int* pivotSize)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	int c = color[cIdx];
	if(c < 0) return;
	// le SCC non sono considerate

	int size = ( OutNodes[v+1] - OutNodes[v] ) * ( InNodes[v+1] - InNodes[v] );
	if(size > pivotSize[c])
	{
		pivotSize[c] = size;
		pivot[c] = v;
	}
}

__global__ void ParSetColor(int V, color_t* color, int* OutNodes, int* InNodes, int* pivot, int* pivotSize, uint8_t *State)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	int c = color[cIdx];
	if(c < 0) return;

	// le SCC non sono considerate

	int size = ( OutNodes[v+1] - OutNodes[v] ) * ( InNodes[v+1] - InNodes[v] );
	if(size > pivotSize[c])
	{
		pivotSize[c] = size;
		pivot[c] = v;
	}
}

__global__ void ParSetPivot(int C, color_t* color, int* pivot, int* globalCounter, int* frontier, int V)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if(c >= C) return;
	int v = pivot[c];
	if(v < 0) return;

	int idx = atomicAdd(globalCounter, 1);
	frontier[idx] = v;

	pivot[c] = -1;

	int cNewIdx = ( v << 1 ) | 1;
	color[cNewIdx] = (v+1)+V*3;
}

__global__ void ParSetPivot(int C, color_t* color, int* pivot, int* globalCounter, uint8_t* State, int V)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if(c >= C) return;
	int v = pivot[c];
	if(v < 0) return;

	atomicAdd(globalCounter, 1);
	int cNewIdx = ( v << 1 ) | 1;

	pivot[c]     = -1;
	color[cNewIdx] = (v+1);

	uint8_t s = 0;
	setForwardVisitedBit(&s);
	setBackwardVisitedBit(&s);
	State[v] = s;
}


__global__ void ParSetPivot(int C, color_t* color, int* pivot, int* pivotSize, int* globalCounter, int* frontier, int V)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if(c >= C) return;
	int v = pivot[c];
	if(v < 0) return;

	int idx = atomicAdd(globalCounter, 1);
	frontier[idx] = v;

	pivot[c]     = -1;
	pivotSize[c] = -1;

	int cNewIdx = ( v << 1 ) | 1;
	color[cNewIdx] = (v+1)+V*3;
}

__global__ void ParSetPivot(int C, color_t* color, int* pivot, int* pivotSize, int* globalCounter, uint8_t* State, int V)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if(c >= C) return;
	int v = pivot[c];
	if(v < 0) return;

	atomicAdd(globalCounter, 1);
	int cNewIdx = ( v << 1 ) | 1;

	pivot[c]     = -1;
	pivotSize[c] = -1;
	color[cNewIdx] = (v+1);

	uint8_t s = 0;
	setForwardVisitedBit(&s);
	setBackwardVisitedBit(&s);
	State[v] = s;
}

__global__ void ParTrim(int V, int *OutEdges, int *OutNodes,
												int *InEdges,  int *InNodes,
												color_t* Color,    int *hasToContinue, uint8_t* State)
{
	int v = blockIdx.x*blockDim.x+threadIdx.x;
	if(v >= V) return;

	int cIdx    = ( v << 1 );
	int cNewIdx = ( v << 1 ) | 1;

	color_t c;
	if(isRangeSet(State[v])) return;

	bool hasNeighbor = false;
	// if( Out-degree(n,Color) = 0)
	for (int i = OutNodes[v]; i < OutNodes[v + 1]; ++i) {
		int dest = OutEdges[i];
		//int c2 = Color[dest << 1];

		//if(Color[dest] == Color[qNode]) {
		/*if(c2 == c)
		{
			hasNeighbor = true;
			break;
		}*/
		if(!isRangeSet(State[dest]))
		{
			hasNeighbor = true;
			break;
		}

	}
	// Or ( In-degree(n,Color) = 0)
	if( hasNeighbor )
	{
		hasNeighbor = false;
		for (int i = InNodes[v]; i < InNodes[v + 1]; ++i) {
			int dest = InEdges[i];
			/*int c2 = Color[dest << 1];
			//if(Color[dest] == Color[qNode]) {
			if(c2 == c)
			{
				hasNeighbor = true;
				break;
			}*/
			if(!isRangeSet(State[dest]))
			{
				hasNeighbor = true;
				break;
			}
		}
	}

	if(!hasNeighbor)
	{
		//nessun vicino con lo stesso colore: nuova SCC

		//Color(n) <- -1
		c = -v-1;
		Color[cIdx]    = c;
		Color[cNewIdx] = c;
		rangeSet(&State[v]);
		// SCC <- SCC U {{n}}
		// ancora da pensare a come eseguire questo passaggio...o conviene ricostruirlo lato CPU sfruttando indice in Color?

		// mark{n} <- true

		//this for "until color not changed"
		*hasToContinue = 1;
	}
}


/******************************
 FUNZIONI KERNEL PER IL coloring
 **********************************/
template <bool inverted, bool setColorIntoNegative>
__global__ void ParSetColorToNode(int V, color_t* Color, int* hasToContinue, uint8_t* State)
{
	int v = blockIdx.x*blockDim.x+threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	int cNewIdx = ( v << 1 ) | 1;

	if(setColorIntoNegative || Color[cIdx] >= 0)
	{
		color_t c;
		if(!inverted)
			c = v+1;
		else
			c= V-v; // id al contrario

		Color[cIdx] = c;
		Color[cNewIdx] = c;
		State[v] = 0;
		*hasToContinue = true;
	}
}

template <bool inverted, bool setColorIntoNegative>
__global__ void ParSetColorToNode(int V, color_t* Color, int* hasToContinue)
{
	int v = blockIdx.x*blockDim.x+threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	int cNewIdx = ( v << 1 ) | 1;

	if(setColorIntoNegative || Color[cIdx] >= 0)
	{
		color_t c;
		if(!inverted)
			c = v+1;
		else
			c= V-v; // id al contrario

		Color[cIdx] = c;
		Color[cNewIdx] = c;
		*hasToContinue = true;
	}
}

template <bool inverted>
__global__ void ParSetColorPivot(int V, color_t* Color, int* frontier, int* num_of_scc)
{
	int v = blockIdx.x*blockDim.x+threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	int cNewIdx = ( v << 1 ) | 1;
	color_t c = Color[cIdx];
	if(!inverted)
	{
		if(c == v+1)
		{
			int idx = atomicAdd(num_of_scc, 1);
			frontier[idx] = v;

			c = - c;
			Color[cIdx] = c;
			Color[cNewIdx] = c;
		}
	}
	else
	{
		if(c == V-v)
		{
			int idx = atomicAdd(num_of_scc, 1);
			frontier[idx] = v;

			c = - c;
			Color[cIdx] = c;
			Color[cNewIdx] = c;
		}
	}
}

template <bool inverted>
__global__ void ParSetColorPivot(int V, color_t* Color, int* frontier, int* num_of_scc, uint8_t *State)
{
	int v = blockIdx.x*blockDim.x+threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	color_t c = Color[cIdx];
	if(!inverted)
	{
		if(c == v+1)
		{
			*num_of_scc = 1;

			uint8_t s = 0;
			setBackwardVisitedBit(&s);
			State[v] = s;
		}
	}
	else
	{
		if(c == V-v)
		{
			*num_of_scc = 1;

			uint8_t s = 0;
			setBackwardVisitedBit(&s);
			State[v] = s;
		}
	}
}

__global__ void count_scc(int V, color_t* color, int *num_of_scc)
{
	int v = blockIdx.x*blockDim.x+threadIdx.x;
	if(v >= V) return;
	if(color[v<<1] < 0) atomicAdd(num_of_scc, 1);
}

__global__ void FwdMaxColor(int V,
					int *InEdges, int *InNodes,
					color_t* Color, uint8_t *State, int* hasToContinue
					)
{
	int v = blockIdx.x*blockDim.x+threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	int cNewIdx = ( v << 1 ) | 1;
	uint8_t s = State[v];
	if(isRangeSet(s)) return;

	color_t c = Color[cIdx];


	// iterate over neighbors
	int num_nbr = InNodes[v+1] - InNodes[v];
	int* nbrs = & InEdges[ InNodes[v] ];

  int oldColor = c;
	for(int i = 0; i < num_nbr; i++) {
		int w = nbrs[i];
		color_t c2 = Color[w << 1];
		//if(c2 < 0) continue;
		//c = max(atomicMax(&Color[v], c2), c);
		c = max(c, c2);
		//if (c < c2) {
			//*hasToContinue = true;
			//Color[v] = c2;
			//c  = c2;//aggiorna info colore locale
		//}
	}
	if( c != oldColor)
	{
		*hasToContinue = true;
		Color[cIdx] = c;
		Color[cNewIdx] = c;
	}
}

__global__ void ResetState(int V, uint8_t *State)
{
	int v = blockIdx.x*blockDim.x+threadIdx.x;
	if(v >= V) return;
	if(!isRangeSet(State[v])) State[v] = 0;
}

__global__ void FwdMaxColor(int V,
					int *InEdges, int *InNodes,
					color_t* Color, int* hasToContinue
					)
{
	int v = blockIdx.x*blockDim.x+threadIdx.x;
	if(v >= V) return;
	int cIdx = ( v << 1 );
	int cNewIdx = ( v << 1 ) | 1;
	color_t c = Color[cIdx];
	if(c < 0) return;

	// iterate over neighbors
	int num_nbr = InNodes[v+1] - InNodes[v];
	int* nbrs = & InEdges[ InNodes[v] ];

  int oldColor = c;
	for(int i = 0; i < num_nbr; i++) {
		int w = nbrs[i];
		color_t c2 = Color[w << 1];
		//if(c2 < 0) continue;
		//c = max(atomicMax(&Color[v], c2), c);
		c = max(c, c2);
		//if (c < c2) {
			//*hasToContinue = true;
			//Color[v] = c2;
			//c  = c2;//aggiorna info colore locale
		//}
	}
	if( c != oldColor)
	{
		*hasToContinue = true;
		Color[cIdx] = c;
		Color[cNewIdx] = c;
	}
}

template<unsigned BLOCK_DIM, unsigned WARP_SIZE, VisitType visitType>
__global__ void QuadraticBBFS(	int* __restrict__ devNodes,
										int* __restrict__ devEdges,
										color_t* __restrict__ Color,
										uint8_t* __restrict__ State,
										const int V,
										const int level) {

	const int ID = (blockIdx.x * BLOCK_DIM + threadIdx.x);
	const int VirtualID =  (blockIdx.x * BLOCK_DIM + threadIdx.x) >> _Log2<WARP_SIZE>::VALUE;
	if (ID == 0)
		devF2Size[(level & 3) + 1] = false;
	bool change = false;

	int cIdx    = ( VirtualID << 1 );
	int cNewIdx = ( VirtualID << 1 ) | 1;


	const int laneID = threadIdx.x & _Mod2<WARP_SIZE>::VALUE;
	//const int Stride = (gridDim.x * BLOCK_DIM) >> _Log2<WARP_SIZE>::VALUE;
	const int i = VirtualID;
	//for (int i = VirtualID; i < V; i += Stride) {
		uint8_t s = State[i];
		bool hasToVisit = isBackwardVisited(s) && !isBackwardPropagate(s) && !isRangeSet(s);

		if ( hasToVisit ) {
			color_t cSrc = Color[cIdx];
			color_t cNew = 0;
			if(visitType == SCC_Decomposition) cNew = Color[cNewIdx];

			const int start = devNodes[i];
			const int end = devNodes[i + 1];
			uint8_t nbrState;

			for (int j = start + laneID; j < end; j += WARP_SIZE) {
				const int dest = devEdges[j];
				nbrState = State[dest];

				if (!isRangeSet(nbrState) && !isBackwardVisited(nbrState) && Color[dest<<1] == cSrc)
				{
					if(visitType == SCC_Decomposition) Color[( dest << 1 ) | 1] = cNew;
					setBackwardVisitedBit(&State[dest]);
					change = true;
				}
			}

			if(visitType == Coloring && laneID == 0)
			{
				//need to update color!
				rangeSet(&State[i]);
				Color[cIdx] = -cSrc;
				Color[cNewIdx] = -cSrc;
			}
			else
				setBackwardPropagateBit(&State[i]);
		}//hasToVisit
	//}
	if (__any(change))
		devF2Size[level & 3] = 1;
}

template<unsigned BLOCK_DIM, unsigned WARP_SIZE, VisitType visitType>
__global__ void QuadraticFBFS(	int* __restrict__ devNodes,
										int* __restrict__ devEdges,
										color_t* __restrict__ Color,
										uint8_t* __restrict__ State,
										const int V,
										const int level) {

	const int ID = (blockIdx.x * BLOCK_DIM + threadIdx.x);
	const int VirtualID =  (blockIdx.x * BLOCK_DIM + threadIdx.x) >> _Log2<WARP_SIZE>::VALUE;
	if (ID == 0)
		devF2Size[(level & 3) + 1] = false;
	bool change = false;

	int cIdx    = ( VirtualID << 1 );
	int cNewIdx = ( VirtualID << 1 ) | 1;

	const int laneID = threadIdx.x & _Mod2<WARP_SIZE>::VALUE;
	//const int Stride = (gridDim.x * BLOCK_DIM) >> _Log2<WARP_SIZE>::VALUE;
	const int i = VirtualID;
	//for (int i = VirtualID; i < V; i += Stride) {
		uint8_t s = State[i];
		bool hasToVisit = isForwardVisited(s) && !isForwardPropagate(s) && !isRangeSet(s);

		if ( hasToVisit ) {
			color_t cSrc = Color[cIdx];
			int cNew = 0;
			if(visitType == SCC_Decomposition) cNew = Color[cNewIdx];

			const int start = devNodes[i];
			const int end = devNodes[i + 1];
			uint8_t nbrState;

			for (int j = start + laneID; j < end; j += WARP_SIZE) {
				const int dest = devEdges[j];
				nbrState = State[dest];

				if (!isRangeSet(nbrState) && !isForwardVisited(nbrState) && Color[dest << 1] == cSrc)
				{
						if(visitType == SCC_Decomposition) Color[( dest << 1 ) | 1] = cNew;
						setForwardVisitedBit(&State[dest]);
						change = true;
				}
			}

			setForwardPropagateBit(&State[i]);
		}//hasToVisit
	//}
	if (__any(change))
		devF2Size[level & 3] = 1;
}

//#define fun(a)		BFS_KernelMainGLOB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE>\
//								<<<DIV(FrontierSize, (BLOCKDIM / (a)) * ITEM_PER_WARP), BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
//								(devOutNodes, devOutEdges, devDistance, devF1, devF2, FrontierSize, level);

#define fun(a, f, visitType, dynPar)		BFS_KernelMainGLOB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE, f, visitType, dynPar>\
<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
(V, devOutNodes, devOutEdges, devDistance, devF1, devF2, devColor, devState, FrontierSize, level);

#define funB(a, f, visitType, dynPar)		BFS_KernelMainGLOBB	<BLOCKDIM, (a), false, DUPLICATE_REMOVE, f, visitType, dynPar>\
<<<gridDim, BLOCKDIM, SMem_Per_Block(BLOCKDIM)>>>\
(V, devOutNodes, devOutEdges, devDistance, devF1, devF2, devColor, devState, FrontierSize, level);
template<bool forward, VisitType visitType, int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, int WARP_SIZE>
inline int cudaGraph::closure(int* devOutNodes, int* devOutEdges, dist_t* devDistance, int FrontierSize)
{
	int SizeArray[4];
	int level = 1;
	int totalNodes = FrontierSize;
	while ( FrontierSize ) {
		FrontierDebug(FrontierSize, level, PRINT_FRONTIER);
		int size = logValueHost<MIN_WARP, MAX_WARP>(FrontierSize);

		const int DynBlocks = DYNAMIC_PARALLELISM ? RESERVED_BLOCKS : 0;
		const int gridDim = min(SCC_MAX_CONCURR_TH/BLOCKDIM - DynBlocks, DIV(FrontierSize, BLOCKDIM));

		if(WARP_SIZE == WARP_SIZE_MAGIC_VALUE)
		{
			if (size >= 3 && FrontierSize > SCC_MAX_CONCURR_TH) {
				def_SWITCHB(size, forward, visitType, DYNAMIC_PARALLELISM);
			} else {
				def_SWITCH(size, forward, visitType, DYNAMIC_PARALLELISM);
			}
		}
		else
		{

			if(forward)
				QuadraticFBFS<BLOCKDIM, WARP_SIZE, visitType><<<DIV(V*WARP_SIZE, BLOCKDIM), BLOCKDIM>>>(devOutNodes, devOutEdges, devColor, devState, V, level);
			else
				QuadraticBBFS<BLOCKDIM, WARP_SIZE, visitType><<<DIV(V*WARP_SIZE, BLOCKDIM), BLOCKDIM>>>(devOutNodes, devOutEdges, devColor, devState, V, level);
		}


		//low difference, but kernel behave better
		//copyColor<256, MAX_CONCURR_BL(256)><<<MAX_CONCURR_BL(256), 256>>>(V, devColorToTake, devColor, MAX_CONCURR_BL(256));
		//copyColor2<256><<<MAX_CONCURR_BL(256), 256>>>(V, devColorToTake, devColor);
		//cudaMemcpy(devColorToTake, devColor, V * sizeof(int), cudaMemcpyDeviceToDevice);


		//cudaMemcpyFromSymbolAsync(SizeArray, devF2Size, sizeof(int) * 4);
		cudaMemcpyFromSymbol(SizeArray, devF2Size, sizeof(int) * 4);
		FrontierSize = SizeArray[level & 3];
		if (FrontierSize > this->allocFrontierSize)
		error("BFS Frontier too large. Required more GPU memory. N. of Vertices/Edges in frontier: " << FrontierSize << " >  allocated: " << this->allocFrontierSize);

		std::swap<int*>(devF1, devF2);
		level++;
		totalNodes += FrontierSize;
	}

	return totalNodes;
}

template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, int WARP_SIZE, bool COLOR_FWD>
inline int cudaGraph::coloring(std::deque<LogValues>& loggedValues, LogValues& value_to_log)
{
	const bool inverted = false;
	int totalKernelCalled = 0;

	bool hasToContinue = false;
	int no_of_scc = 0;

	int* devNodes = devOutNodes;
	int* devEdges = devOutEdges;

	if(COLOR_FWD)
	{
		devNodes = devOutNodes;
		devEdges = devOutEdges;
	}
	else
	{
		devNodes = devInNodes;
		devEdges = devInEdges;
	}

	cudaMemset(D_hasToContinue, 0x00, sizeof(int));
	if(WARP_SIZE == WARP_SIZE_MAGIC_VALUE)
		ParSetColorToNode<inverted, false><<<DIV(V, BLOCKDIM), BLOCKDIM>>>(graph.v(), devColor, D_hasToContinue);
	else
		ParSetColorToNode<inverted, false><<<DIV(V, BLOCKDIM), BLOCKDIM>>>(graph.v(), devColor, D_hasToContinue, devState);

	cudaMemcpy( &hasToContinue, D_hasToContinue, sizeof(int),  cudaMemcpyDeviceToHost );

	if(hasToContinue)
	{
		do{
			cudaMemset(D_hasToContinue, 0x00, sizeof(int));//false
			if(WARP_SIZE == WARP_SIZE_MAGIC_VALUE)
				FwdMaxColor<<<DIV(graph.v(), BLOCKDIM), BLOCKDIM>>>(graph.v(), devEdges, devNodes, devColor, D_hasToContinue);
			else
				FwdMaxColor<<<DIV(graph.v(), BLOCKDIM), BLOCKDIM>>>(graph.v(), devEdges, devNodes, devColor, devState, D_hasToContinue);
			cudaMemcpy( &hasToContinue, D_hasToContinue, sizeof(int),  cudaMemcpyDeviceToHost );
		}while(hasToContinue);

		cudaMemset(D_num_of_scc, 0x00, sizeof(int));//false
		if(WARP_SIZE == WARP_SIZE_MAGIC_VALUE)
			ParSetColorPivot<inverted><<<DIV(V, BLOCKDIM), BLOCKDIM>>>(graph.v(), devColor, devF1, D_num_of_scc);
		else
			ParSetColorPivot<inverted><<<DIV(V, BLOCKDIM), BLOCKDIM>>>(graph.v(), devColor, devF1, D_num_of_scc, devState);
		cudaMemcpy( &no_of_scc, D_num_of_scc, sizeof(int),  cudaMemcpyDeviceToHost );

		//int SizeArray[4] = {no_of_scc, 0, 0, 0};
		int SizeArray[4] = {0, 0, 0, 0};
		cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);

		//std::cout << "Recusion level " << value_to_log.recursionLevel << " - SCC " << no_of_scc << std::endl;
		int no_of_vertex_found = closure<false, Coloring, MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, WARP_SIZE>(devNodes, devEdges, devDistanceIn, no_of_scc);
	}

	return no_of_scc;
}

template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE>
inline int cudaGraph::fwbw(std::deque<LogValues>& loggedValues, LogValues& value_to_log)
{
	int FrontierSize = 0;
	int SizeArray[4] = {0, 0, 0, 0};
	value_to_log.subRecursionLevel = 0;

	// Selezione dei pivot!

	CUDA_ERROR("FWBW - before pivot devGlobalCounter");

	cudaMemset(devGlobalCounter, 0, sizeof(int));
	CUDA_ERROR("FWBW - after pivot devGlobalCounter");
	if(EURISTIC)
	{
		if(WARP_SIZE == WARP_SIZE_MAGIC_VALUE)
		{
			ParSetColor<<<DIV(V, BLOCKDIM), BLOCKDIM>>>(V, devColor, devOutNodes, devInNodes, devPivotPerColor, devPivotPerColorSize);
			CUDA_ERROR("FWBW - after pivot ParSetColor");
			ParSetPivot<<<DIV(3*V+1, BLOCKDIM), BLOCKDIM>>>(V*3+1, devColor, devPivotPerColor, devPivotPerColorSize, devGlobalCounter, devF1, V);
			CUDA_ERROR("FWBW - after pivot ParSetPivot");
		}
		else
		{
			ParSetColor<<<DIV(V, BLOCKDIM), BLOCKDIM>>>(V, devColor, devOutNodes, devInNodes, devPivotPerColor, devPivotPerColorSize, devState);
			CUDA_ERROR("FWBW - after pivot ParSetColor");
			ParSetPivot<<<DIV(3*V+1, BLOCKDIM), BLOCKDIM>>>(V*3+1, devColor, devPivotPerColor, devPivotPerColorSize, devGlobalCounter, devState, V);
			CUDA_ERROR("FWBW - after pivot ParSetPivot");
		}
	}
	else
	{
		if(WARP_SIZE == WARP_SIZE_MAGIC_VALUE)
		{
			ParSetColor<<<DIV(V, BLOCKDIM), BLOCKDIM>>>(V, devColor, devPivotPerColor);
			ParSetPivot<<<DIV(3*V+1, BLOCKDIM), BLOCKDIM>>>(V*3+1, devColor, devPivotPerColor, devGlobalCounter, devF1, V);
		}
		else
		{
			ParSetColor<<<DIV(V, BLOCKDIM), BLOCKDIM>>>(V, devColor, devPivotPerColor, devState);
			ParSetPivot<<<DIV(3*V+1, BLOCKDIM), BLOCKDIM>>>(V*3+1, devColor, devPivotPerColor, devGlobalCounter, devState, V);
		}
	}
	//if(FrontierSize >= V) std::cout << "ERRORE: frontiera più grande di V " << std::endl;
	cudaMemcpy(&FrontierSize, devGlobalCounter, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(frontier, devF1, FrontierSize * sizeof(int), cudaMemcpyDeviceToHost);

	CUDA_ERROR("FWBW - after pivot");

	// nessun nodo pivot!
	if(FrontierSize == 0) return 0;

	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);
	int nodesForward = closure<true, SCC_Decomposition, MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, WARP_SIZE>(devOutNodes, devOutEdges, devDistance, FrontierSize);

	CUDA_ERROR("FWBW - closure fw");

	cudaMemcpy(devF1, frontier, FrontierSize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);

	int nodesBackward = closure<false, SCC_Decomposition, MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, WARP_SIZE>(devInNodes, devInEdges, devDistanceIn, FrontierSize);

	CUDA_ERROR("FWBW - closure bw");

	if(WARP_SIZE == WARP_SIZE_MAGIC_VALUE)
		updateColor<<<DIV(V, BLOCKDIM), BLOCKDIM>>>(V, devState, devColor);
	else
		updateColor<<<DIV(V, BLOCKDIM), BLOCKDIM>>>(V, devColor, devState);

	return FrontierSize;

}

int cudaGraph::trim1(std::deque<LogValues>& loggedValues, LogValues& value_to_log)
{
	int n = 0;
	cudaMemset(devGlobalCounter, 0, sizeof(int));
	ParTrim<<<DIV(V, BLOCKDIM), BLOCKDIM>>>(V, devOutEdges, devOutNodes, devInEdges, devInNodes, devColor, devGlobalCounter, devState);
	cudaMemcpy(&n, devGlobalCounter, 1 * sizeof(int), cudaMemcpyDeviceToHost);

	return n;
}

template <class charT, charT sep>
class punct_facet: public std::numpunct<charT>{
protected:
    charT do_decimal_point() const { return sep; }
};

inline bool cudaGraph::cudaBFS4K(int* source) {
	return false;
}

template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD>
bool cudaGraph::cudaSCC4K(int maxTrimIteration, int totalFWBWIteration, int idRun){
	if (DUPLICATE_REMOVE)
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	else
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault);

	LogValues value_to_log;
	std::deque<LogValues> loggedValues;
	timeout = false;

	bool brutal_end = false;

	if (INTER_BLOCK_SYNC) {
		//BFSDispath<256, DUPLICATE_REMOVE, SCC_Decomposition> <<<MAX_CONCURR_BL(256), 256, SMem_Per_Block(256)>>>
		//		(V, devOutNodes, devOutEdges, devDistance, devF1, devF2, devColor, devState);

		return false;
	}
	else {
		CUDA_ERROR("BEGIN_PSCC");
		int n = 1;
		int FrontierSize = 1;

		value_to_log.recursionLevel = 0;
		value_to_log.subRecursionLevel = 0;

		int totalTrimIteration = 0;
		n = 1;
		while(n > 0 && (maxTrimIteration == -1 || totalTrimIteration < maxTrimIteration))
		{
			++totalTrimIteration;
			n = trim1(loggedValues, value_to_log);
		}
		CUDA_ERROR("TRIM");


		do
		{
			value_to_log.recursionLevel++;

			if(value_to_log.recursionLevel <= totalFWBWIteration)
			{
				FrontierSize = fwbw<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW>(loggedValues, value_to_log);
				CUDA_ERROR("FWBW");
			}
			else
			{
				FrontierSize = coloring<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, WARP_SIZE_COL, COLOR_FWD>(loggedValues, value_to_log);
				CUDA_ERROR("COLORING");
			}
		}while(FrontierSize != 0);

		color_t* cSA = new color_t[V*2];
		cudaMemcpy(cSA, devColor, 2*graph.v() * sizeof(color_t), cudaMemcpyDeviceToHost);
		int scc = 0;

		for(int i = 0; i < 2*V; i+=2)
		{
			if(abs(cSA[i]) == (i/2+1))
				scc++;
		}

		delete[] cSA;
	}
	globalErr = false;
	return true;
}

#undef fun

void cudaGraph::Reset(const int Sources[], int nof_sources) {
	cudaError("Graph Reset src");
	this->Reset();
	
	cudaMemcpy(devF1, Sources, nof_sources * sizeof(int), cudaMemcpyHostToDevice);
	cudaUtil::scatterKernel<dist_t><<<DIV(nof_sources, 128), 128>>>(devF1, nof_sources, devDistance, 0);
		cudaError("End Graph Reset src");
}

void cudaGraph::Reset() {
	cudaError("Graph Reset");
	cudaUtil::fillKernel<dist_t><<<DIV(V, 128), 128>>>(devDistance, V, INF );
	cudaUtil::fillKernel<dist_t><<<DIV(V, 128), 128>>>(devDistanceIn, V, INF );

	//SCC
	//cudaMemset(devColor, 0, V * sizeof(color_t));
	color_t c_base;
	c_base = 0;
	cudaUtil::fillKernel<color_t><<<DIV(2*V, 128), 128>>>(devColor, 2*V, c_base );

	//cudaUtil::fillKernel<int><<<DIV(V, 128), 128>>>(devColor, V, 0 );
	//cudaUtil::fillKernel<int><<<DIV(V, 128), 128>>>(devColorToTakeF, V, 0 );
	//cudaUtil::fillKernel<int><<<DIV(V, 128), 128>>>(devColorToTakeB, V, 0 );
	cudaUtil::fillKernel<int><<<DIV(V*3+1, 128), 128>>>(devPivotPerColor, V*3+1, -1 );
	cudaUtil::fillKernel<int><<<DIV(V*3+1, 128), 128>>>(devPivotPerColorSize, V*3+1, -1 );
	cudaUtil::fillKernel<uint8_t><<<DIV(V, 128), 128>>>(devState, V, 0 );

	int SizeArray[4] = {0, 0, 0, 0};
	cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);

	GReset<<<1, 256>>>();
	cudaError("Graph Reset");
}


// ---------------------- AUXILARY FUNCTION ---------------------------------------------

inline void cudaGraph::FrontierDebug(int FrontierSize, int level, bool PRINT_F) {
	totalFrontierNodes += FrontierSize;
	if (PRINT_F == 0)
		return;
	std::stringstream ss;
	ss << "Level: " << level << "\tF2Size: " << FrontierSize << std::endl;
	if (PRINT_F == 2)
		printExt::printCudaArray(devF1, FrontierSize, ss.str());
}

template<int MIN_VALUE, int MAX_VALUE>
inline int cudaGraph::logValueHost(int Value) {
	int logSize = 31 - __builtin_clz(SCC_MAX_CONCURR_TH / Value);
	if (logSize < _Log2<MIN_VALUE>::VALUE)
		logSize = _Log2<MIN_VALUE>::VALUE;
	if (logSize > _Log2<MAX_VALUE>::VALUE)
		logSize = _Log2<MAX_VALUE>::VALUE;
	return logSize;
}

		/*if (BLOCK_BFS && FrontierSize < BLOCK_FRONTIER_LIMIT) {
			BFS_BlockKernel<DUPLICATE_REMOVE><<<1, 1024, 49152>>>(devNodes, devEdges, devDistance, devF1, devF2, FrontierSize);
			cudaMemcpyFromSymbolAsync(&level, devLevel, sizeof(int));
		} else {*/

		/*template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM>
		inline bool cudaGraph::cudaSCC4K(int DUPLICATE_REMOVE, int switchColoring)
		{
			if (DUPLICATE_REMOVE)
				cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
			switch(DUPLICATE_REMOVE)
			{
				case 0: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, 0>(switchColoring);
				case 1: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, 1>(switchColoring);
				default: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, 0>(switchColoring);
			}
		}
		template<int MIN_WARP, int MAX_WARP>
		inline bool cudaGraph::cudaSCC4K(bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, int switchColoring)
		{
			switch(DYNAMIC_PARALLELISM)
			{
				case true: return cudaSCC4K<MIN_WARP, MAX_WARP, true>(DUPLICATE_REMOVE, switchColoring);
				case false: return cudaSCC4K<MIN_WARP, MAX_WARP, false>(DUPLICATE_REMOVE, switchColoring);
				//can't be!
				default: return false;
			}
		}*/

		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL>
		bool cudaGraph::cudaSCC4K(bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int idRun)
		{
			switch(COLOR_FWD)
			{
				case true: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, true>(maxTrimIteration, totalFWBWIteration, idRun);
				case false: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, false>(maxTrimIteration, totalFWBWIteration, idRun);

				//default: 2
				default: std::cout << "Ignored case - COLOR_FWD" << std::endl; return false;// cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, false>(maxTrimIteration, totalFWBWIteration, idRun);
			}
		}

		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW>
		bool cudaGraph::cudaSCC4K(int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int idRun)
		{
			switch(WARP_SIZE_COL)
			{
				case 0: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, 1024>(COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 1: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, 1>(COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				case 2: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, 2>(COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				case 4: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, 4>(COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 8: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, 8>(COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 16: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, 16>(COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 32: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, 32>(COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);

				//default: 2
				default: std::cout << "Ignored case - WARP_SIZE_COL" << std::endl; return false;//cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, 2>(COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
			}
		}

		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC>
		bool cudaGraph::cudaSCC4K(int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int idRun)
		{
			switch(WARP_SIZE_FWBW)
			{
				case 0: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, 1024>(WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 1: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, 1>(WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 2: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, 2>(WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				case 4: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, 4>(WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				case 8: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, 8>(WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 16: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, 16>(WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 32: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, 32>(WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);

				//default: 2
				default: std::cout << "Ignored case - WARP_SIZE_FWBW" << std::endl; return false;//cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, 2>(WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
			}
		}

		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE>
		bool cudaGraph::cudaSCC4K(bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int idRun)
		{
			switch(EURISTIC)
			{
				case true: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, true>(WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case false: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, false>(WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				default: std::cout << "Ignored case - EURISTIC" << std::endl; return false;//cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, false>(WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
			}
		}

		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM>
		bool cudaGraph::cudaSCC4K(int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int idRun)
		{
			switch(DUPLICATE_REMOVE)
			{
				case 0: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, 0>(EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 1: return cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, 1>(EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				default: std::cout << "Ignored case - DUPLICATE_REMOVE" << std::endl; return false;//cudaSCC4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, 0>(EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
			}
		}
		template<int MIN_WARP, int MAX_WARP>
		bool cudaGraph::cudaSCC4K(bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int idRun)
		{
			switch(DYNAMIC_PARALLELISM)
			{
				//case true: return cudaSCC4K<MIN_WARP, MAX_WARP, true>(DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				case false: return cudaSCC4K<MIN_WARP, MAX_WARP, false>(DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//can't be!
				default: std::cout << "Ignored case - DYNAMIC_PARALLELISM" << std::endl; return false;
			}
		}

		template<int MIN_WARP>
		bool cudaGraph::cudaSCC4K(int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int idRun)
		{
			switch(MAX_WARP)
			{
				/*case 1: return cudaSCC4K<MIN_WARP, 1>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, maxTrimIteration, totalFWBWIteration, idRun);
				case 2: return cudaSCC4K<MIN_WARP, 2>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, maxTrimIteration, totalFWBWIteration, idRun);
				case 4: return cudaSCC4K<MIN_WARP, 4>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, maxTrimIteration, totalFWBWIteration, idRun);
				case 8: return cudaSCC4K<MIN_WARP, 8>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, maxTrimIteration, totalFWBWIteration, idRun);
				case 16: return cudaSCC4K<MIN_WARP, 16>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, maxTrimIteration, totalFWBWIteration, idRun);
				case 32: return cudaSCC4K<MIN_WARP, 32>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, maxTrimIteration, totalFWBWIteration, idRun);*/

				default: return cudaSCC4K<MIN_WARP, 32>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
			}
		}

		bool cudaGraph::cudaSCC4K(int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE_FWBW, int WARP_SIZE_COL, bool COLOR_FWD, int maxTrimIteration, int totalFWBWIteration, int idRun)
		{
			switch(MIN_WARP)
			{
				//case 1: return cudaSCC4K<1>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				//case 2: return cudaSCC4K<2>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				case 4: return cudaSCC4K<4>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				case 8: return cudaSCC4K<8>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				case 16: return cudaSCC4K<16>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
				case 32: return cudaSCC4K<32>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);

				default: std::cout << "Ignored case - MIN_WARP" << std::endl; return false;//cudaSCC4K<4>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE_FWBW, WARP_SIZE_COL, COLOR_FWD, maxTrimIteration, totalFWBWIteration, idRun);
			}
		}
		
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE>
		bool cudaGraph::cudaBFS4K(int num_of_sources)//no source, had to call that BEFORE using RESET!
		{
			int FrontierSize = num_of_sources;
			
			int SizeArray[4] = {0, 0, 0, 0};
			cudaMemcpyToSymbol(devF2Size, SizeArray, sizeof(int) * 4);

			// start from that ID!
			//cudaMemcpy(&source, devF1, sizeof(int), cudaMemcpyDeviceToHost);

			CUDA_ERROR("BFS - after frontier");

			//closure<true, BFS, MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, WARP_SIZE>(devOutNodes, devOutEdges, devDistance, FrontierSize);
			closure<true, BFS, MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, WARP_SIZE>(devOutNodes, devOutEdges, devDistance, FrontierSize);
			CUDA_ERROR("BFS - fw");
			
			return true;
		}
		
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC>
		bool cudaGraph::cudaBFS4K(int WARP_SIZE, int source)
		{
			switch(WARP_SIZE)
			{
				case 0: return cudaBFS4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, 1024>(source);

				default: std::cout << "Ignored case - WARP_SIZE" << std::endl; return false;
			}
		}
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE>
		bool cudaGraph::cudaBFS4K(bool EURISTIC, int WARP_SIZE, int source)
		{
			switch(EURISTIC)
			{
				case true: return cudaBFS4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, true>(WARP_SIZE, source);
				//case false: return cudaBFS4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, false>(WARP_SIZE, source);
				
				default: std::cout << "Ignored case - EURISTIC" << std::endl; return false;
			}
		}
		
		template<int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM>
		bool cudaGraph::cudaBFS4K(int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE, int source)
		{
			switch(DUPLICATE_REMOVE)
			{
				//case 1: return cudaBFS4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, 1>(EURISTIC, WARP_SIZE, source);
				case 0:  return cudaBFS4K<MIN_WARP, MAX_WARP, DYNAMIC_PARALLELISM, 0>(EURISTIC, WARP_SIZE, source);

				default: std::cout << "Ignored case - DUPLICATE_REMOVE" << std::endl; return false;
			}
		}
		
		template<int MIN_WARP, int MAX_WARP>
		bool cudaGraph::cudaBFS4K(bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE, int source)
		{
			switch(DYNAMIC_PARALLELISM)
			{
				case false: return cudaBFS4K<MIN_WARP, MAX_WARP, false>(DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				//case true:  return cudaBFS4K<MIN_WARP, MAX_WARP, true>(DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source); //<-- THIS LINE CAUSE A BUG!

				default: std::cout << "Ignored case - DYNAMIC_PARALLELISM" << std::endl; return false;
			}
		}
		
		template<int MIN_WARP>
		bool cudaGraph::cudaBFS4K(int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE, int source)
		{
			switch(MAX_WARP)
			{
				//case 1: return cudaBFS4K<MIN_WARP, 1>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				//case 2: return cudaBFS4K<MIN_WARP, 2>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				//case 4: return cudaBFS4K<MIN_WARP, 4>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				//case 8: return cudaBFS4K<MIN_WARP, 8>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				//case 16: return cudaBFS4K<MIN_WARP, 16>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				case 32: return cudaBFS4K<MIN_WARP, 32>(DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);

				default: std::cout << "Ignored case - MAX_WARP" << std::endl; return false;
			}
		}
		
		bool cudaGraph::cudaBFS4K(int MIN_WARP, int MAX_WARP, bool DYNAMIC_PARALLELISM, int DUPLICATE_REMOVE, bool EURISTIC, int WARP_SIZE, int source)
		{
			switch(MIN_WARP)
			{
				//case 1: return cudaBFS4K<1>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				//case 2: return cudaBFS4K<2>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				case 4: return cudaBFS4K<4>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				case 8: return cudaBFS4K<8>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				case 16: return cudaBFS4K<16>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);
				case 32: return cudaBFS4K<32>(MAX_WARP, DYNAMIC_PARALLELISM, DUPLICATE_REMOVE, EURISTIC, WARP_SIZE, source);

				default: std::cout << "Ignored case - MIN_WARP" << std::endl; return false;
			}
		}
}
