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
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>

int main() {
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	std::cout << std::endl << "Number of Streaming Multiprocessors:\t" << devProp.multiProcessorCount
	 		  << std::endl << "                 Compute Cabability:\t" << devProp.major << devProp.minor << '0'
			  << std::endl << std::endl;

	if (std::getenv("NUM_OF_SM") == NULL) {
		std::ofstream file;
		try {
	    	file.open(std::string(std::getenv("HOME")).append("/.bashrc").c_str(), std::fstream::app);
	  	}
	  	catch (std::ios_base::failure &fail) {
	    	std::cout << "An exception occurred: bashrc not found"  << std::endl;
	  	}

		file << std::endl << "export NUM_OF_SM=" << devProp.multiProcessorCount
			 << std::endl << "export CUDA_ARCH=" << devProp.major << devProp.minor << "0" << std::endl;
		file.close();

		std::cout << "please exec:   source ~/.bashrc" << std::endl << std::endl;
	}
}
