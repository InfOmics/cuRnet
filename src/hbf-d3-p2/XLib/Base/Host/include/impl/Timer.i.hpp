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
#include <iomanip>				// set precision cout
#include <chrono>
#include <ratio>
#include "../fUtil.hpp"

namespace std {
    template<class Rep, std::intmax_t Num, std::intmax_t Denom>
    std::ostream& operator<<(std::ostream& os, __attribute__((unused)) const std::chrono::duration<Rep, std::ratio<Num, Denom>>& ratio) {
        if (Num == 3600 && Denom == 1)		return os << " h";
        else if (Num == 60 && Denom == 1)	return os << " min";
        else if (Num == 1 && Denom == 1)	return os << " s";
        else if (Num == 1 && Denom == 1000)	return os << " ms";
        else return os << " Unsupported";
    }
}

namespace timer {
    using float_seconds = std::chrono::duration<float>;

	template<typename>
	struct is_duration : std::false_type {};

	template<typename T, typename R>
	struct is_duration<std::chrono::duration<T, R>> : std::true_type {};

//-------------------------- GENERIC -------------------------------------------

	template<timerType type>
	Timer<type>::Timer(std::ostream& _outStream, int _decimals) : outStream(_outStream), decimals(_decimals) {}

#if defined(__COLOR)
	template<timerType type>
	Timer<type>::Timer(int _decimals, int _space, StreamModifier::Code _color) :
	 					 decimals(_decimals), space(_space), defaultColor(_color)  {}
#else
    template<timerType type>
	Timer<type>::Timer(int _decimals, int _space) :
	 					 decimals(_decimals), space(_space)  {}
#endif

    template<timerType type>
	Timer<type>::~Timer() {}

	template<timerType type>
	template<typename _ChronoPrecision>
	void Timer<type>::print(std::string str) {
		static_assert(is_duration<_ChronoPrecision>::value, "Wrong type : typename is not std::chrono::duration");
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
		static_assert(is_duration<_ChronoPrecision>::value, "Wrong type : typename is not std::chrono::duration");
		this->stop();
		this->print(str);
	}

//-------------------------- HOST ----------------------------------------------

	template<>
	template<typename _ChronoPrecision>
	float Timer<HOST>::duration() {
		static_assert(is_duration<_ChronoPrecision>::value, "Wrong type : typename is not std::chrono::duration");
		return std::chrono::duration_cast<_ChronoPrecision>(endTime - startTime).count();
	}

//-------------------------- CPU -----------------------------------------------
#if defined(__linux__)

	template<>
	template<typename _ChronoPrecision>
	float Timer<CPU>::duration() {
		static_assert(is_duration<_ChronoPrecision>::value, "Wrong type : typename is not std::chrono::duration");
		return std::chrono::duration_cast<_ChronoPrecision>( float_seconds((float) (c_end - c_start) / CLOCKS_PER_SEC) ).count();
	}

//-------------------------- SYS -----------------------------------------------

	template<>
	template<typename _ChronoPrecision>
	float Timer<SYS>::duration() {
		throw std::runtime_error( "Timer<SYS>::duration() is unsupported" );
	}

	template<>
	template<typename _ChronoPrecision>
	void Timer<SYS>::print(std::string str) {
		static_assert(is_duration<_ChronoPrecision>::value, "Wrong type : typename is not std::chrono::duration");
		auto wall_time = std::chrono::duration_cast<_ChronoPrecision>( endTime - startTime ).count();
		auto user_time = std::chrono::duration_cast<_ChronoPrecision>( float_seconds( (float) (endTMS.tms_utime - startTMS.tms_utime) / ::sysconf(_SC_CLK_TCK) ) ).count();
		auto sys_time = std::chrono::duration_cast<_ChronoPrecision>( float_seconds( (float) (endTMS.tms_stime - startTMS.tms_stime) / ::sysconf(_SC_CLK_TCK) ) ).count();

		std::cout __ENABLE_COLOR(<< defaultColor)
                  << std::setw(space) << str
				  << "  Elapsed time: [user " << user_time << ", system " << sys_time << ", real "
				  << wall_time << " " << _ChronoPrecision() << "]"
                  __ENABLE_COLOR(<< StreamModifier::FG_DEFAULT)
                  << std::endl;
	}
#endif
} //@timer
