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
#include <iostream>
#include <string>
#include <nvToolsExt.h>
#include "../include/cuda_util.cuh"

namespace cuda_util {

void memCheckCUDA(size_t Req) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free > Req)
        __ERROR("MEMORY TOO LOW. Req: " << (float) Req / (1<<20) <<
                " MB, available: " << (float) Req / (1<<20) << " MB");
}

bool memInfoCUDA(size_t Req) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "    Total Device Memory:\t" << (total >> 20) << " MB"
              << std::endl
              << "     Free Device Memory:\t" << (free >> 20) << " MB"
              << std::endl
              << "Requested Device memory:\t" << (Req >> 20) << " MB"
              << "\t(" << ((Req >> 20) * 100) / (total >> 20) << "%)"
              << std::endl  << std::endl;
    return free > Req;
}

int deviceProperty::NUM_OF_STREAMING_MULTIPROCESSOR = 0;

int deviceProperty::getNum_of_SMs() {
	if(NUM_OF_STREAMING_MULTIPROCESSOR == 0) {
		cudaDeviceProp devProperty;
		cudaGetDeviceProperties(&devProperty, 0);
		NUM_OF_STREAMING_MULTIPROCESSOR = devProperty.multiProcessorCount;
	}
	return NUM_OF_STREAMING_MULTIPROCESSOR;
}

namespace NVTX {
	/*void PushRange(std::string s, const int color) {
		nvtxEventAttributes_t eventAttrib = {};
		eventAttrib.version = NVTX_VERSION;
		eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		eventAttrib.colorType = NVTX_COLOR_ARGB;
		eventAttrib.color = color; //colors[color_id];
		eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
		eventAttrib.message.ascii = s.c_str();
		nvtxRangePushEx(&eventAttrib);
	}

	void PopRange() {
		nvtxRangePop();
	}*/
}

void cudaStatics() {
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	__CUDA_ERROR("statics");

	std::cout << std::endl
		<< "\t  Graphic Card: " << devProp.name << " (cc: " << devProp.major
                                << "." << devProp.minor << ")" << std::endl
		<< "\t          # SM: " << devProp.multiProcessorCount
		<< "\t  Threads per SM: " << devProp.maxThreadsPerMultiProcessor
		<< "\t    Max Resident Thread: " << devProp.multiProcessorCount *
                                devProp.maxThreadsPerMultiProcessor << std::endl
		<< "\t   Global Mem.: " << devProp.totalGlobalMem / (1 << 20) << " MB"
		<< "\t     Shared Mem.: " << devProp.sharedMemPerBlock / 1024 << " KB"
		<< "\t               L2 Cache: " << devProp.l2CacheSize / 1024
                                                        << " KB" << std::endl
		<< "\tsmemPerThreads: " << devProp.sharedMemPerBlock /
                                 devProp.maxThreadsPerMultiProcessor << " Byte"
		<< "\t  regsPerThreads: " << devProp.regsPerBlock /
                                     devProp.maxThreadsPerMultiProcessor
		<< "\t              regsPerSM: " << devProp.regsPerBlock << std::endl
        << std::endl;

	#if defined(SM)
	if (devProp.multiProcessorCount != SM)
		__ERROR("Wrong SM configuration: " << devProp.multiProcessorCount
                                           << " vs. " << SM)
	#endif
}

} // @CudaUtil
