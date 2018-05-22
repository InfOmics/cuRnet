#pragma once

//extern __shared__ unsigned char SMem[];
#include "../Util/ptx.cu"
#include "../Util/GlobalWrite.cu"


namespace scc4k{
extern __shared__ unsigned char SMem[];

#define PRIMEQ	2654435769u				// ((sqrt(5)-1)/2) << 32

template<int BlockDim>
__device__ __forceinline__ void DuplicateRemove(const int dest, volatile long long int* HashTable, int* Queue, int &founds) {
	unsigned hash = dest * PRIMEQ;
	hash = (hash & (unsigned)(SMem_Per_Block(BlockDim)/12 - 1)) + (hash & (unsigned)(SMem_Per_Block(BlockDim)/24 - 1));

	int2 toWrite = make_int2(Tid, dest);
	HashTable[hash] = reinterpret_cast<volatile long long int&>(toWrite);
	int2 recover = reinterpret_cast<int2*>( const_cast<long long int*>(HashTable) )[hash];
	if (recover.x == Tid || recover.y != dest)
		Queue[founds++] = dest;
	else if (COUNT_DUP && recover.x != Tid && recover.y == dest)
		atomicAdd(&duplicateCounter, 1);
}


template<int BlockDim, bool DUP_REM, bool forward, VisitType visitType>
__device__ __forceinline__ void KVisit( const int V,	const int dest, dist_t* devDistance,
										int* Queue, int& founds, const int level, volatile long long int* HashTable,
										color_t* color, const int colorToConfront, const int colorToTakeValue)
{
	color_t c = color[ (dest << 1) | 1];
	switch(visitType)
	{
	case SCC_Decomposition:
		if(forward)
		{
			if (c == colorToConfront) {
				color[ (dest << 1) | 1 ] = colorToTakeValue;
				//devDistance[dest] = (dist_t) level;

				if (DUP_REM)
					DuplicateRemove<BlockDim>(dest, HashTable, Queue, founds);
				else
					Queue[founds++] = dest;
			}
		}
		else
		{
			bool upd = false;
			if (c   == colorToConfront) { color[ (dest << 1) | 1 ] = colorToTakeValue;   upd = true;} // not visited forward
			if (c+V == colorToTakeValue) { color[ (dest << 1) | 1 ] = colorToTakeValue+V; upd = true;} //visited forward

			if(upd ) {
				//color[ (dest << 1) | 1 ] = c + colorToTakeValue;
				//devDistance[dest] = (dist_t) level;

				if (DUP_REM)
					DuplicateRemove<BlockDim>(dest, HashTable, Queue, founds);
				else
					Queue[founds++] = dest;
			}
		}
	break;
	case Coloring:
		//if( c3.color < 0 ) break;//non processare elementi già processati

		if (c >= 0 && -c == colorToConfront) {
			color[ dest << 1 ] = -c;
			color[ ( dest << 1 ) | 1 ] = -c;

			if (DUP_REM)
				DuplicateRemove<BlockDim>(dest, HashTable, Queue, founds);
			else
				Queue[founds++] = dest;
		}
	case BFS:
		if( devDistance[dest] == INF )
		{
			devDistance[dest] = level;
			Queue[founds++] = dest;
		}
		break;
	}
}


template<int BlockDim, int WARP_SZ, bool DUP_REM, bool forward, VisitType visitType>
__device__ __forceinline__ void EdgeVisit(const int V,	 	   int* __restrict__	devEdge,
												dist_t* __restrict__	devDistance,
												   int* __restrict__ 	devF2,
												   int* __restrict__	devF2SizePrt,
													int start, int end,
													int* Queue, int& founds, const int level, volatile long long int* HashTable,
													color_t* __restrict__ color, const int colorToConfront, const int colorToTakeValue) {
#if SAFE == 0
	for (int k = start + (Tid & _Mod2<WARP_SZ>::VALUE); k < end; k += WARP_SZ) {
		const int dest = devEdge[k];

		KVisit<BlockDim, DUP_REM, forward, visitType>(V, dest, devDistance, Queue, founds, level, HashTable, color, colorToConfront, colorToTakeValue);
	}
#elif SAFE == 1
	bool flag = true;
	int k = start + (Tid & _Mod2<WARP_SZ>::VALUE);
	while (flag) {
		while (k < end && founds < REG_QUEUE) {
			const int dest = devEdge[k];

			KVisit<BlockDim, DUP_REM, forward, visitType>(V, dest, devDistance, Queue, founds, level, HashTable, color, colorToConfront, colorToTakeValue);
			k += WARP_SZ;
		}
		if (__any(founds >= REG_QUEUE)) {
			FrontierWrite::Write<BlockDim, FrontierWrite::SIMPLE>(devF2, devF2SizePrt, Queue, founds);
			founds = 0;
		} else
			flag = false;
	}
#endif
}


}
