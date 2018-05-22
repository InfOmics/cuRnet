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

#if defined(__COLOR)
    #include "fUtil.hpp"
#endif

#if defined(__linux__)
	#include <ctime>		//CPU
	#include <sys/times.h>	//SYS
#endif

namespace timer {

enum timerType {  HOST					// Wall (real) clock host time
			#if defined(__linux__)
				, CPU					// User time
				, SYS					// User/Kernel/System time
			#endif
};


#if defined(__COLOR)
    #define __ENABLE_COLOR(...)  __VA_ARGS__
#else
    #define __ENABLE_COLOR(...)
#endif
/**
* @class Timer
* @brief Timer class for HOST and DEVICE
* HOST timer: "HOST" (default) Wall (real) clock host time, "CPU" User time, "SYS" User/Kernel/System time
* "DEVICE" Wall clock device time
*/
template<timerType type>
class Timer {

protected:
	int decimals;
	int space;
    __ENABLE_COLOR(StreamModifier::Code defaultColor;)
	std::ostream& outStream = std::cout;
private:
	// HOST
	std::chrono::system_clock::time_point startTime, endTime;
#if defined(__linux__)
	// CPU
	std::clock_t c_start, c_end;
	// SYS
	struct tms startTMS, endTMS;
#endif

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
	virtual void start();

	/** Stop the timer */
	virtual void stop();

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
};

}

#include "impl/Timer.i.hpp"
