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
#include <exception>
#include <algorithm>
#include <ratio>

namespace numeric {

	namespace {
		#if !NO_CONSTEXPR
		constexpr int  BinCoeffAux(const int x, const int y)	{ return x <= y ? 1 : x * BinCoeffAux(x - 1, y);}

		template<unsigned V, int i>
		constexpr unsigned nearestPower2Aux1() {
			return i == 32 ? V + 1 : nearestPower2Aux1<V | (V >> i), i * 2>();
		}
		#endif

		template<int X, int Y>
		struct BINCOEFF_AUX {
			static const int value = X * BINCOEFF_AUX<X - 1, Y - 1>::value;
		};

		template<int X>
		struct BINCOEFF_AUX<X, 1> {
			static const int value = 1;
		};
	}

	#if !NO_CONSTEXPR
		template<typename T, typename R>
		constexpr T    Min(const T a, const R b)				{ return a < b ? a : b; 							}
		template<typename T, typename R>
		constexpr T    Max(const T a, const R b)				{ return a > b ? b : a; 							}
		constexpr int  Div(const int n, const int div)			{ return (n + div - 1) / div;						}
		constexpr int  UpperApprox(const int n, const int MUL)	{ return Div(n, MUL) * MUL;							}
		constexpr int  LowerApprox(const int n, const int MUL)	{ return (n / MUL) * MUL;							}

		constexpr bool CheckPow2(const int x)					{ return (x != 0) && !(x & (x - 1));				}
		constexpr int  FACTORIAL(const int x)					{ return x <= 1 ? 1 : x * FACTORIAL(x - 1);			}
		constexpr int  BinCoeff(const int x, const int y)		{ return BinCoeffAux(x, x - y) / ( FACTORIAL(y) );	}
		constexpr int  Mod2(const int X, const int MOD)     	{ return X & (MOD - 1);								}

		template<int MUL>
		constexpr int NearestMul2(const int n) {
			static_assert(CheckPow2(MUL), "Not power of 2");
			return n & ~((1 << LOG2<MUL>::value) - 1);
		}

		template<unsigned V>
		constexpr unsigned _nearestPower2() {
			return nearestPower2Aux1<V - 1, 5>();
		}
	#endif

    //--------------------------------------------------------------------------

	template<int A, int B> struct MAX {	static const int value = A > B ? A : B; };
	template<int A, int B> struct MIN {	static const int value = A < B ? A : B;	};

    //lower bound
	template<int N>
	struct LOG2 {
        static_assert(N > 0, "LOG2 : N <= 0");
        static_assert(IS_POWER2<N>::value, "LOG2 : N is not power of two");

		static const int value = 1 + LOG2<N / 2>::value;
	};
	template<>	struct LOG2<1> { static const int value = 0;	};

	template<int N>
	struct MOD2 {
		static_assert(N > 0, "MOD2 : N <= 0");
        static_assert(IS_POWER2<N>::value, "MOD2 : N is not power of two");

		static const int value = N - 1;
	};

    template<int N>
	struct IS_POWER2 {
		static const bool value = (N != 0) && !(N & (N - 1));
	};

	/*template<int N>
	int UpperApproxPow2 {
		static_assert(N > 0, "log2");
		return
	}*/


	template<int N>
	struct FACTORIAL {
		static_assert(N >= 0, "FACTORIAL");
		static const int value = N * FACTORIAL<N - 1>::value;
	};

	template<> struct FACTORIAL<0> { static const int value = 1; };

	template<int N, int K>
	struct BINCOEFF {
		static_assert(N >= 0 && K >= 0 && N >= K, "FACTORIAL");
		static const int value = BINCOEFF<N - 1, K - 1>::value + BINCOEFF<N - 1, K>::value;
	};
	template<>
	struct BINCOEFF<0, 0> {		static const int value = 1;						};
	template<int N>
	struct BINCOEFF<N, 0> {		static const int value = 1;						};
	template<int N>
	struct BINCOEFF<N, N> {		static const int value = 1;						};
	/*template<int N, int K>
	struct BINCOEFF_1 {
		static const int value = K > N - K ? (BINCOEFF_AUX<N, N - K>::value / FACTORIAL<N - K>::value) : (BINCOEFF_AUX<N, K>::value / FACTORIAL<K>::value);
	};*/


    inline int nearestPower2_UP(int v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

    inline int log2(const int v) {
        return 31 - __builtin_clz(v);
    }

	template<class T>
	float average(T* Input, const int size) {
		const T sum = std::accumulate(Input, Input + size, 0);
		return (float) sum / size;
	}

	template<class T>
	float stdDeviation(T* Input, const int size, float avg) {
		float sum = 0;
		for (int i = 0; i < size; ++i)
			sum += std::pow(Input[i] - avg, 2);
		return std::sqrt(sum / size);
	}

	template<class T>
	float stdDeviation(T* Input, const int size) {
		const float avg = average(Input, size);
		return stdDeviation(Input, size, avg);
	}

	template<typename T>
	float perCent(const T part, const T max) {
		return ((float) part / max) * 100;
	}

    template<std::intmax_t Num, std::intmax_t Den>
    struct compareFloatABS_Str<std::ratio<Num, Den>> {
        template<typename T>
        inline bool operator() (const T a, const T b) {
            const T epsilon = static_cast<T>(Num) / static_cast<T>(Den);
            return std::abs(a - b) < epsilon;
        }
    };

    template<std::intmax_t Num, std::intmax_t Den>
    struct compareFloatRel_Str<std::ratio<Num, Den>> {
        template<typename T>
        inline bool operator() (const T a, const T b) {
            const T epsilon = static_cast<T>(Num) / static_cast<T>(Den);
            const T diff = std::abs(a - b);
            //return (diff < epsilon) || (diff / std::max(std::abs(a), std::abs(b)) < epsilon);
            return (diff < epsilon) ||
            (diff / std::min(std::abs(a) + std::abs(b), std::numeric_limits<float>::max()) < epsilon);
        }
    };
} //@numeric
