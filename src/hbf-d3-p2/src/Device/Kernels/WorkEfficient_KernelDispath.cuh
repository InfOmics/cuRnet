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
#pragma once

#include "Device/WorkEfficient_KernelFun.cuh"
using namespace kernels;

#define __GLOBAL_DEVICE__ __global__
#define NAME1 BF_Kernel1

#include "WorkEfficient_KernelMain.cuh"

#undef __GLOBAL_DEVICE__
#undef NAME1

#define __GLOBAL_DEVICE__ __device__ __forceinline__
#define NAME1 BF_KernelD1

#include "WorkEfficient_KernelMain.cuh"

#undef __GLOBAL_DEVICE__
#undef NAME1

__global__ void BFSDispath(	edge_t* __restrict__ devNodes,
							EdgeT* __restrict__ devEdges,
							hdist_t* __restrict__ devDistances,
							node_t* devF1,
							node_t* devF2,
							long long unsigned* __restrict__ d_distances) {

	int devF1SizeLocal = 1;
	int level = 1;
	int* devF2SizePrt = devF2Size + (level & 3);

	while (devF1SizeLocal) {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            devF2Size[(level + 1) & 3] = 0;

        int size = basic::log2(RESIDENT_THREADS / devF1SizeLocal);
    	if (MIN_VW >= 1 && size < LOG2<MIN_VW>::value)
    		size = LOG2<MIN_VW>::value;
    	if (MAX_VW >= 1 && size > LOG2<MAX_VW>::value)
    		size = LOG2<MAX_VW>::value;

        #define fun(a)	BF_KernelD1<a, true>(devNodes, devEdges, devDistances,\
                                             devF1, devF2, devF1SizeLocal, level, d_distances);

        def_SWITCH(size);

        #undef fun

		global_sync::GlobalSync<BLOCKDIM>();

		devF1SizeLocal = devF2SizePrt[0];
		level++;
		devF2SizePrt = devF2Size + (level & 3);
        basic::swap(devF1, devF2);
	}
}
