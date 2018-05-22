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
#include <iomanip>				// set precision cout
#include <ratio>

namespace timer_cuda {

template<timerType type>
Timer<type>::Timer(std::ostream& _outStream, int _decimals) :
            timer::Timer<timer::timerType::HOST>(_outStream, _decimals) {}

#if defined(__COLOR)

template<timerType type>
Timer<type>::Timer(int _decimals, int _space, StreamModifier::Code _color) :
 					 timer::Timer<timer::timerType::HOST>(_decimals, _space, _color) {
	cudaEventCreate(&startTimeCuda);
	cudaEventCreate(&stopTimeCuda);
}

#else

template<timerType type>
Timer<type>::Timer(int _decimals, int _space) :
 					 timer::Timer<timer::timerType::HOST>(_decimals, _space) {
	cudaEventCreate(&startTimeCuda);
	cudaEventCreate(&stopTimeCuda);
}

#endif


template<timerType type>
Timer<type>::~Timer() {
    cudaEventDestroy( startTimeCuda );
    cudaEventDestroy( stopTimeCuda );
}

template<timerType type>
template<typename _ChronoPrecision>
void Timer<type>::print(std::string str) {
    static_assert(timer::is_duration<_ChronoPrecision>::value, "Wrong type : typename is not std::chrono::duration");
    std::cout __ENABLE_COLOR(<< this->defaultColor)
              << std::setw(this->space) << str << '\t'
              << std::fixed << std::setprecision(this->decimals) << this->duration<_ChronoPrecision>()
              << _ChronoPrecision()
              __ENABLE_COLOR(<< StreamModifier::FG_DEFAULT)
              << std::endl << std::endl;
}

template<timerType type>
template<typename _ChronoPrecision>
void Timer<type>::getTime(std::string str) {
    static_assert(timer::is_duration<_ChronoPrecision>::value, "Wrong type : typename is not std::chrono::duration");
    this->stop();
    this->print(str);
}

//-------------------------- HOST ----------------------------------------------

template<>
template<typename _ChronoPrecision>
float Timer<DEVICE>::duration() {
	static_assert(timer::is_duration<_ChronoPrecision>::value, "Wrong type : typename is not std::chrono::duration");
	using float_milliseconds = typename std::chrono::duration<float, std::milli>;
	float time;
	cudaEventElapsedTime(&time, startTimeCuda, stopTimeCuda);
	return std::chrono::duration_cast<_ChronoPrecision>( float_milliseconds( time )).count();
}

template<>
template<typename _ChronoPrecision>
void Timer<DEVICE>::getTimeError(std::string str, const char* file, int line) {
	static_assert(timer::is_duration<_ChronoPrecision>::value, "Wrong type : typename is not std::chrono::duration");
	getTime<_ChronoPrecision>(str);
	cudaDeviceSynchronize();
	cuda_util::__getLastCudaError(str.c_str(), file, line);
}

} //@timer
