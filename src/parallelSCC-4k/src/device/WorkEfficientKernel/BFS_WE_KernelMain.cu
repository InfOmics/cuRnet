namespace scc4k{

template<int BlockDim, int WARP_SZ, bool SYNC, bool DUP_REM, bool forward, VisitType visitType, bool DYNAMIC_PARALLELISM>
__GLOBAL_DEVICE__ void NAME1 (int V,  int* __restrict__		devNode,
								     int* __restrict__		devEdge,
								  dist_t* __restrict__ 		devDistance,
								     int* __restrict__ 		devF1,
								     int* __restrict__	 	devF2,
										 color_t* __restrict__ devColor,
										 uint8_t* __restrict__	 	devState,
								const int devF1Size, const int level) {

	int* devF2SizePrt = &devF2Size[level & 3];	// mod 4
	if (blockIdx.x == 0 && Tid == 0)
		devF2Size[(level + 1) & 3] = 0;

	volatile long long int* HashTable;
	if (DUP_REM)
		HashTable = (volatile long long int*) SMem;

	int founds = 0;
	int Queue[REG_QUEUE];

	const int VirtualID = (blockIdx.x * BlockDim + Tid) >> _Log2<WARP_SZ>::VALUE;
	const int Stride =  gridDim.x * (BlockDim >> _Log2<WARP_SZ>::VALUE);

	const int size = ceilf(__fdividef(devF1Size, gridDim.x));
	const int maxLoop = (size + BlockDim / WARP_SZ - 1) >> (_Log2<BlockDim>::VALUE - _Log2<WARP_SZ>::VALUE);

	for (int t = VirtualID, loop = 0; loop < maxLoop; t += Stride, loop++) {
		int index = 0, start = 0, end = 0, colorToConfront = 0, colorToTakeValue = 0;
		if (t < devF1Size) {
			index = devF1[t];
			start = devNode[index];
			end = devNode[index + 1];
			
			if(visitType != BFS)
			{
				colorToConfront  = devColor[index << 1];
				colorToTakeValue = devColor[( index << 1 ) | 1];

				if(forward)
				{
					if(colorToTakeValue > 3*V) colorToTakeValue = colorToTakeValue - (V<<1);//- V*2 -->prende ID+V;
				}
				else
				{
					if(colorToTakeValue > 3*V) colorToTakeValue = colorToTakeValue - V;//- V -->prende ID+V*2;
				}
			}

			DynamicParallelism<BlockDim, WARP_SZ, 0, forward, visitType, DYNAMIC_PARALLELISM>(V, devEdge, devDistance, devColor, devF2, start, end, level, colorToConfront, colorToTakeValue);
		} else
			end = INT_MIN;

		EdgeVisit<BlockDim, WARP_SZ, DUP_REM, forward, visitType>(V, devEdge, devDistance, devF2, devF2SizePrt, start, end, Queue, founds, level, HashTable, devColor, colorToConfront, colorToTakeValue);
	}
	if (DUP_REM && (STORE_MODE == FrontierWrite::SHARED_WARP || STORE_MODE == FrontierWrite::SHARED_BLOCK))
		__syncthreads();
	FrontierWrite::Write<BlockDim, FrontierWrite::SHARED_WARP>(devF2, devF2SizePrt, Queue, founds);
}



template<int BlockDim, int WARP_SZ, bool SYNC, bool DUP_REM, bool forward, VisitType visitType, bool DYNAMIC_PARALLELISM>
__GLOBAL_DEVICE__ void NAME1B (	int V, 	 int* __restrict__		devNode,
								     int* __restrict__		 devEdge,
								  dist_t* __restrict__ 		 devDistance,
								     int* __restrict__ 		 devF1,
								     int* __restrict__	 	 devF2,
										 color_t* __restrict__ devColor,
										 uint8_t* __restrict__	 	 devState,
								const int devF1Size, const int level) {

	int* devF2SizePrt = &devF2Size[level & 3];	// mod 4
	if (blockIdx.x == 0 && Tid == 0)
		devF2Size[(level + 1) & 3] = 0;

	volatile long long int* HashTable;
	if (DUP_REM)
		HashTable = (volatile long long int*) SMem;

	int* SH_start = (int*) SMem;
	int* SH_end = ((int*) SMem) + BlockDim;
	int* SH_colorConfront = ((int*) SMem) + BlockDim*2;
	int* SH_colorToTake = ((int*) SMem) + BlockDim*3;

	const int VWarpID = Tid >> _Log2<WARP_SZ>::VALUE;
	SH_start += VWarpID * WARP_SZ;
	SH_end += VWarpID * WARP_SZ;
	SH_colorConfront += VWarpID * WARP_SZ;
	SH_colorToTake   += VWarpID * WARP_SZ;

	int founds = 0;
	int Queue[REG_QUEUE];
	const int VlocalID = Tid & _Mod2<WARP_SZ>::VALUE;
	const int CSIZE = WARP_SZ > 32 ? BlockDim : 32;

	const int size = (devF1Size + CSIZE - 1) & ~(CSIZE - 1);   // ((devF1Size + 32 - 1) / 32) * 32;
	const int Stride = gridDim.x * BlockDim;

	for (int ID = blockIdx.x * BlockDim + Tid; ID < size; ID += Stride) {
		int index = 0, Tstart = 0, Tend = 0, TColorConfront = 0, TColorToTake = 0;
		if (ID < devF1Size) {
			index = devF1[ID];
			Tstart = devNode[index];
			Tend = devNode[index + 1];
			
			if(visitType != BFS)
			{
				TColorConfront = devColor[index << 1];
				TColorToTake   = devColor[( index << 1 ) | 1];
				if(forward)
				{
					if(TColorToTake > 3*V) TColorToTake = TColorToTake - (V<<1);//- V*2 -->take ID+V;
				}
				else
				{
					if(TColorToTake > 3*V) TColorToTake = TColorToTake - V;//- V -->take ID+V*2;
				}
			}

			DynamicParallelism<BlockDim, WARP_SZ, 1, forward, visitType, DYNAMIC_PARALLELISM>(V, devEdge, devDistance, devColor, devF2, Tstart, Tend, level, TColorConfront, TColorToTake);
		} else
			Tend = INT_MIN;

		if (WARP_SZ > 32) {
			__syncthreads();
			SH_start[VlocalID] = Tstart;
			SH_end[VlocalID] = Tend;
			SH_colorConfront[VlocalID] = TColorConfront;
			SH_colorToTake[VlocalID] = TColorToTake;
			__syncthreads();
			for (int i = 0; i < WARP_SZ; i++) {
				int start = SH_start[i];
				int end = SH_end[i];
				if(visitType != BFS)
				{
					int colorToConfront = SH_colorConfront[i];
					int colorToTake     = SH_colorToTake[i];

					EdgeVisit<BlockDim, WARP_SZ, DUP_REM, forward, visitType>(V, devEdge, devDistance, devF2, devF2SizePrt, start, end, Queue, founds, level, HashTable, devColor, colorToConfront, colorToTake);
				}
				else
				{
					EdgeVisit<BlockDim, WARP_SZ, DUP_REM, forward, visitType>(V, devEdge, devDistance, devF2, devF2SizePrt, start, end, Queue, founds, level, HashTable, devColor, 0, 0);
				}
			}
		}
		else {
			for (int i = 0; i < WARP_SZ; i++) {
				int start = __shfl(Tstart, i, WARP_SZ);
				int end = __shfl(Tend, i, WARP_SZ);
				if(visitType != BFS)
				{
					int colorToConfront = __shfl(TColorConfront, i, WARP_SZ);
					int colorToTake = __shfl(TColorToTake, i, WARP_SZ);
					EdgeVisit<BlockDim, WARP_SZ, DUP_REM, forward, visitType>(V, devEdge, devDistance, devF2, devF2SizePrt, start, end, Queue, founds, level, HashTable, devColor, colorToConfront, colorToTake);
				}
				else
				{
					EdgeVisit<BlockDim, WARP_SZ, DUP_REM, forward, visitType>(V, devEdge, devDistance, devF2, devF2SizePrt, start, end, Queue, founds, level, HashTable, devColor, 0, 0);
				}
			}
		}
	}

	if (DUP_REM && (STORE_MODE == FrontierWrite::SHARED_WARP || STORE_MODE == FrontierWrite::SHARED_BLOCK))
		__syncthreads();
	FrontierWrite::Write<BlockDim, FrontierWrite::SIMPLE>(devF2, devF2SizePrt, Queue, founds);
}

}
