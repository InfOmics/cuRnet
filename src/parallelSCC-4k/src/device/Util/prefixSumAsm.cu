#pragma once

#include "ptx.cu"

namespace scc4k{

template<int VW_SIZE>
__device__ __forceinline__ void warpInclusiveScanAsm(int& x) {

	if (VW_SIZE == 32) {
		asm(
			"{.reg .s32 r0;"
			".reg .s32 r1;"
			".reg .pred p;"
			"mov.s32 r0, %1;"
			"shfl.up.b32  r1|p, r0, 0x1,  0x0;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x2,  0x0;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x4,  0x0;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x8,  0x0;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x10, 0x0;"
			"@p add.s32 r0, r1, r0;"
			"mov.s32 %0, r0; }"
			: "=r"(x) : "r"(x));
	}
	if (VW_SIZE == 16) {
		asm(
			"{.reg .s32 r0;"
			".reg .s32 r1;"
			".reg .pred p;"
			"mov.s32 r0, %1;"
			"shfl.up.b32  r1|p, r0, 0x1,  0x1000;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x2,  0x1000;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x4,  0x1000;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x8,  0x1000;"
			"@p add.s32 r0, r1, r0;"
			"mov.s32 %0, r0; }"
			: "=r"(x) : "r"(x));
	}
	if (VW_SIZE == 8) {
		asm(
			"{.reg .s32 r0;"
			".reg .s32 r1;"
			".reg .pred p;"
			"mov.s32 r0, %1;"
			"shfl.up.b32  r1|p, r0, 0x1, 0x1800;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x2, 0x1800;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x4, 0x1800;"
			"@p add.s32 r0, r1, r0;"
			"mov.s32 %0, r0; }"
			: "=r"(x) : "r"(x));
	}
	if (VW_SIZE == 4) {
		asm(
			"{.reg .s32 r0;"
			".reg .s32 r1;"
			".reg .pred p;"
			"mov.s32 r0, %1;"
			"shfl.up.b32  r1|p, r0, 0x1,  0x1C00;"
			"@p add.s32 r0, r1, r0;"
			"shfl.up.b32  r1|p, r0, 0x2,  0x1C00;"
			"@p add.s32 r0, r1, r0;"
			"mov.s32 %0, r0; }"
			: "=r"(x) : "r"(x));
	}

	if (VW_SIZE == 2) {
		asm(
			"{.reg .s32 r0;"
			".reg .s32 r1;"
			".reg .pred p;"
			"mov.s32 r0, %1;"
			"shfl.up.b32  r1|p, r0, 0x1,  0x1E00;"
			"@p add.s32 r0, r1, r0;"
			"mov.s32 %0, r0; }"
			: "=r"(x) : "r"(x));
	}
}


template<int N_OF_VALUES>
__device__ __forceinline__ int warpExclusiveScanBallot(bool Flags[], int& prefixSum) {
	int warpTotal = 0;
	
	#pragma unroll 
	for (int i = 0; i < N_OF_VALUES - 1; i++)
		warpTotal += __popc(__ballot(Flags[i]));

	const unsigned last = __ballot(Flags[N_OF_VALUES - 1]);
	prefixSum = warpTotal + __popc(last & LaneMaskLt());
	
	return warpTotal + __popc(last);
}


__device__ int warpExclusiveScanNoAsm(int& value) {
	const int lane = Tid & 31;

	int n = __shfl_up(value, 1, 32);
	if (lane >= 1)
		value += n;
	n = __shfl_up(value, 2, 32);
	if (lane >= 2)
		value += n;
	n = __shfl_up(value, 4, 32);
	if (lane >= 4)
		value += n;
	n = __shfl_up(value, 8, 32);
	if (lane >= 8)
		value += n;
	n = __shfl_up(value, 16, 32);
	if (lane >= 16)
		value += n;

	const int pSum = __shfl(value, 31, 32); // somma totale di un warp
	value = __shfl_up(value, 1, 32); // Somma prefissa esclusiva
	if (!lane)
		value = 0;
	return pSum;
}


template<int VW_SIZE>
__device__ __forceinline__ int warpExclusiveScan(int& var) {

	warpInclusiveScanAsm<VW_SIZE>(var);

	const int total = __shfl(var, VW_SIZE - 1);
	var = __shfl_up(var, 1);
	if (LaneID() == 0)
		var = 0;
	return total;
}

}
