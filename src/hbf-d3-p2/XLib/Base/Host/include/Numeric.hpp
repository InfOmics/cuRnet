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
#include <numeric>
#include <cmath>

#define NO_CONSTEXPR 1

namespace numeric {
	#if NO_CONSTEXPR
		#define _CheckPow2(x)		((x != 0) && !(x & (x - 1)))
		#define _CheckPow2(x)		((x != 0) && !(x & (x - 1)))

		#define _Div(A, B)			(((A) + (B) - 1) / (B))
		#define _UpperApprox(A, B)	(_Div((A), (B)) * (B))

		#define    _Min(A, B)		(((A) < (B)) ? (A) : (B))
		#define    _Max(A, B)		(((A) > (B)) ? (A) : (B))
	#else
		template<typename T, typename R>
		constexpr T  _Min			(const T a, const R b);
		template<typename T, typename R>
		constexpr T  _Max			(const T a, const R b);
		constexpr int  _Div			(const int n, const int div);
		constexpr int  _UpperApprox	(const int n, const int MUL);
		constexpr int  _LowerApprox	(const int n, const int MUL);

		constexpr bool _CheckPow2	(const int x);
		constexpr int  _Factorial	(const int x);
		constexpr int  _BinCoeff	(const int x, const int y);
		constexpr int  _Mod2		(const int X, const int MOD);
		//constexpr int  log_2		(int n);
	#endif

	template<int N>	struct LOG2;
	template<int N> struct MOD2;
    template<int N> struct IS_POWER2;
	template<int N>	struct FACTORIAL;
	template<int N, int K>	struct BINCOEFF;
	template<int A, int B> struct MAX;
	template<int A, int B> struct MIN;
	template<int N, int MUL, int INDEX = 0>	struct fastStringToIntStr;

	template<int MUL>
	constexpr int NearestMul2(const int n);

    inline int nearestPower2_UP(int v);
    inline int log2(const int v);

	bool isPositiveInteger(const std::string& s);

	template<typename T>
	float perCent(const T part, const T max);

	template<class T>
	float average(T* Input, const int size);

	template<class T>
	float stdDeviation(T* Input, const int size);

    template<typename R>
    struct compareFloatABS_Str {
        template<typename T>
        inline bool operator() (const T a, const T b);
    };

    template<typename R>
    struct compareFloatRel_Str {
        template<typename T>
        inline bool operator() (const T a, const T b);
    };

	inline int fastStringToInt(const char* str, const int len);
}

#include "impl/Numeric.i.hpp"
