#include <cuda_runtime.h>
#include <iostream>
int main(int argc, char* argv[]) {
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	if (argv[1][0] == '0')
		std::cout << devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor;
}
