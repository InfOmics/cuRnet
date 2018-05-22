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

#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include "../../../Host/BaseHost.hpp"

#include "cuda_util.cuh"	// getLastError
#include <cuda_runtime.h>
#define getTimeErr(msg)			getTimeError((msg), __FILE__, __LINE__)

namespace timer_cuda {

enum timerType { DEVICE	};			// Wall (real) clock device time

/**
* @class Timer
* @brief Timer class for HOST and DEVICE
* HOST timer: "HOST" (default) Wall (real) clock host time, "CPU" User time, "SYS" User/Kernel/System time
* "DEVICE" Wall clock device time
*/
template<timerType type>
class Timer : public timer::Timer<timer::timerType::HOST> {
	private:
		// DEVICE
		cudaEvent_t startTimeCuda, stopTimeCuda;

	public:
		/**
		* Default costructor
		*/
#if defined(__COLOR)
        Timer(int _decimals = 1, int _space = 1,
                StreamModifier::Code color = StreamModifier::FG_DEFAULT);
#else
        Timer(int _decimals = 1, int _space = 1);
#endif
		Timer(std::ostream& _outStream, int _decimals = 1);
        ~Timer();

		/** Start the timer */
		void start() override;

		/** Stop the timer */
		void stop() override;

		/*
		* Get the time elapsed between start() and stop()
		* @return time elapsed
		*/
		template<typename _ChronoPrecision = std::chrono::duration<float, std::milli>>
		float duration();

		/*
		* Print the time elapsed between start() and stop()
		* if start() and stop() not invoked indef behavior
		*/
		template<typename _ChronoPrecision = std::chrono::duration<float, std::milli>>
		void print(std::string str);

		/*
		* Stop the timer and print the time elapsed between start() and stop()
		* if start() and stop() not invoked indef behavior
		*/
		template<typename _ChronoPrecision = std::chrono::duration<float, std::milli>>
		void getTime(std::string str);

		/*
		* Stop the timer, check if error and print the time elapsed between start() and stop()
		* if start() and stop() not invoked indef behavior
		*/
		template<typename _ChronoPrecision = std::chrono::duration<float, std::milli>>
		void getTimeError(std::string str, const char* file, int line);
};

}

#include "impl/Timer.i.cuh"
